%==========================================================================
% DRONE IMAGE PROCESSING & TEXTURE SEGMENTATION SCRIPT
%==========================================================================

TEXTURE_CLASSES = {'car', 'grass', 'tree'};
TEXTURE_SAMPLES_DIR = 'texture_samples'; % Folder containing car_01.png, grass_01.png, etc.

% --- Parameters for image processing and segmentation ---
SCALE_FACTOR = 0.25; 
NUM_ITERATIONS = 35; 
VOTING_WINDOW_DIA = 60;

% --- Car Detection Parameters ---
CAR_CLASS_LABEL_IDX = find(strcmp(TEXTURE_CLASSES, 'car'));

% Load data
data = load('drone_images.mat');
num_pairs = size(data.image_pairs, 1);
num_pairs_to_process = 1;

% Process each image pair
for i = 1:num_pairs
    % --- Steps 1-5: Image Restoration ---
    d_color = data.image_pairs{i, 1};
    P_original = data.image_pairs{i, 2};
    d_double = im2double(d_color);
    
    % Step 2: Preprocessing (Scaling and PSF resizing/normalization)
    d_scaled = imresize(d_double, SCALE_FACTOR);
    new_psf_size = round(size(P_original) * SCALE_FACTOR);
    new_psf_size = new_psf_size + 1 - mod(new_psf_size, 2);
    P_scaled = imresize(P_original, new_psf_size(1:2));
    if size(P_scaled, 3) == 1
        P = P_scaled / sum(P_scaled(:));
    else
        P = zeros(size(P_scaled));
        for c = 1:3
            P_channel = P_scaled(:,:,c);
            P(:,:,c) = P_channel / sum(P_channel(:));
        end
    end
    
    % Step 3: Richardson-Lucy Deconvolution
    u_restored_scaled = richardson_lucy_deconv(d_scaled, P, NUM_ITERATIONS);
    
    % Step 4: Post-processing (Upscaling)
    u_restored = imresize(u_restored_scaled, [size(d_color, 1), size(d_color, 2)]);
    
    % Step 5: Wallis Filter for Contrast Enhancement
    WINDOW_SIZE = 51; TARGET_MEAN = 0.5; TARGET_STD = 0.5; AMAX = 5.0; P_CONST = 0.2;
    u_enhanced = wallis_filter(u_restored, WINDOW_SIZE, TARGET_MEAN, TARGET_STD, AMAX, P_CONST);
    
    % --- Step 6: Load Texture Samples from Folder ---
    fprintf('\n=== Loading texture samples from folder ===\n');
    T_cell_all = cell(length(TEXTURE_CLASSES), 1);
    
    for class_idx = 1:length(TEXTURE_CLASSES)
        class_name = TEXTURE_CLASSES{class_idx};
        pattern = fullfile(TEXTURE_SAMPLES_DIR, [class_name, '_*.png']);
        files = dir(pattern);
        
        if isempty(files)
            error('No texture samples found for class "%s" in %s\nExpected pattern: %s_*.png', ...
                  class_name, TEXTURE_SAMPLES_DIR, class_name);
        end
        
        % Load all samples for this class
        numSamples = length(files);
        samples = cell(numSamples, 1);
        
        for s = 1:numSamples
            filepath = fullfile(TEXTURE_SAMPLES_DIR, files(s).name);
            img = imread(filepath);
            
            % Convert to grayscale if needed
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            
            samples{s} = img;
        end
        
        T_cell_all{class_idx} = samples;
        fprintf('  Class %d (%s): %d samples loaded\n', class_idx, class_name, numSamples);
    end
    
    % --- Step 7: Visualize Restoration & Load Summary ---
    figure('Position', [100, 100, 1600, 400]);
    subplot(1, 5, 1); imshow(d_color); title(['Pair ', num2str(i), ': Degraded Image']);
    subplot(1, 5, 2); imshow(P_original, []); title('PSF (Motion Blur)');
    subplot(1, 5, 3); imshow(u_restored); title('R-L Restored');
    subplot(1, 5, 4); imshow(u_enhanced); title('Wallis Enhanced');
    subplot(1, 5, 5); imshow(u_enhanced); title('Texture Samples Loaded');
    
    % --- Step 8: Training Phase (Laws Model with Multiple Samples) ---
    fprintf('\n=== Training Laws model with multiple samples per class ===\n');
    MODEL = training_phase_color_5x5_multi(T_cell_all, laws_kernels_5x5());
    fprintf('Training complete. Model shape: %d classes x %d features (per channel)\n', ...
            size(MODEL.R, 1), size(MODEL.R, 2));
    
    % Convert enhanced image to grayscale for recognition
    I_gray = rgb2gray(u_enhanced);
    
    % --- Step 9: Recognition Phase (Classify Image) ---
    fprintf('\n=== Starting recognition phase on the full image ===\n');
    ClassMap = recognition_phase_color_5x5(u_enhanced, MODEL, laws_kernels_5x5());
    fprintf('Recognition complete.\n');
    
    % --- Step 10: Post-processing (Majority Voting) ---
    fprintf('Applying majority voting filter (window size %d)...\n', VOTING_WINDOW_DIA);
    ClassMap_voted = majority_voting(ClassMap, VOTING_WINDOW_DIA);
    fprintf('Voting complete.\n');
    
    % --- Step 11: Visualize Segmentation Results ---
    figure('Position', [100, 100, 1400, 400]);
    subplot(1, 3, 1);
    imshow(u_enhanced);
    title('Enhanced Image');
    
    subplot(1, 3, 2);
    imshow(ClassMap, [1 length(TEXTURE_CLASSES)]);
    title('Raw Texture Class Map');
    colormap(gca, jet(length(TEXTURE_CLASSES))); 
    c = colorbar; c.Ticks = 1:length(TEXTURE_CLASSES); c.TickLabels = TEXTURE_CLASSES;
    
    subplot(1, 3, 3);
    imshow(ClassMap_voted, [1 length(TEXTURE_CLASSES)]);
    title(sprintf('Majority Voted Map (Window=%d)', VOTING_WINDOW_DIA));
    colormap(gca, jet(length(TEXTURE_CLASSES)));
    c = colorbar; c.Ticks = 1:length(TEXTURE_CLASSES); c.TickLabels = TEXTURE_CLASSES;
    
    % --- Step 12: Car Detection and Visualization ---
    if ~isempty(CAR_CLASS_LABEL_IDX)
        fprintf('\n=== Blob Analysis and Car Detection ===\n');
        stats = detect_and_visualize_cars(ClassMap_voted, u_enhanced, CAR_CLASS_LABEL_IDX);
    else
        fprintf('Could not find "cars" in TEXTURE_CLASSES. Skipping car detection.\n');
    end
end

% =====================================================================
% HELPER FUNCTIONS
% =====================================================================

function Hbank = laws_kernels_5x5()
    % LAWS_KERNELS_5X5
    %   Build 5x5 Laws filter bank
    
    L5 = [ 1  4  6  4  1];
    E5 = [-1 -2  0  2  1];
    S5 = [-1  0  2  0 -1];
    W5 = [-1  2  0 -2  1];
    R5 = [ 1 -4  6 -4  1];
    
    basis = {L5, E5, S5, W5, R5};
    
    Hbank = cell(1, 25);
    idx = 1;
    for i = 1:5
        for j = 1:5
            Hbank{idx} = basis{i}' * basis{j};
            idx = idx + 1;
        end
    end
end

function MODEL = training_phase_color_5x5_multi(T_cell_all, Hbank)
    % TRAINING_PHASE_COLOR_5X5_MULTI
    %   Color training with MULTIPLE SAMPLES per class.
    %   Averages texture descriptors across all samples in each class.
    %
    %   T_cell_all : cell array where each element contains multiple RGB patches
    %   Hbank      : 1 x K cell array of 5x5 kernels
    %
    %   MODEL      : struct with averaged descriptors per class
    
    numClasses = numel(T_cell_all);
    numK = numel(Hbank);
    
    MODEL.R = zeros(numClasses, numK);
    MODEL.G = zeros(numClasses, numK);
    MODEL.B = zeros(numClasses, numK);
    
    for c = 1:numClasses
        samples = T_cell_all{c};
        numSamples = length(samples);
        
        % Accumulate features from all samples
        tempR = zeros(numSamples, numK);
        tempG = zeros(numSamples, numK);
        tempB = zeros(numSamples, numK);
        
        for s = 1:numSamples
            T = im2double(samples{s});
            
            % Handle both grayscale and RGB
            if size(T, 3) == 1
                % Grayscale: replicate to RGB
                T = repmat(T, 1, 1, 3);
            elseif size(T, 3) ~= 3
                error('Expected RGB or grayscale patch in class %d, sample %d', c, s);
            end
            
            Tr = T(:,:,1);
            Tg = T(:,:,2);
            Tb = T(:,:,3);
            
            % Compute Laws energy for each filter
            for k = 1:numK
                H = Hbank{k};
                
                % R channel
                Ar = conv2(Tr, H, 'same');
                tempR(s,k) = sum(Ar(:).^2) / numel(Tr);
                
                % G channel
                Ag = conv2(Tg, H, 'same');
                tempG(s,k) = sum(Ag(:).^2) / numel(Tg);
                
                % B channel
                Ab = conv2(Tb, H, 'same');
                tempB(s,k) = sum(Ab(:).^2) / numel(Tb);
            end
        end
        
        % Average features across all samples for this class
        MODEL.R(c,:) = mean(tempR, 1);
        MODEL.G(c,:) = mean(tempG, 1);
        MODEL.B(c,:) = mean(tempB, 1);
        
        fprintf('  Class %d: Averaged %d samples\n', c, numSamples);
    end
end

function ClassMap = recognition_phase_color_5x5(I_rgb, MODEL, Hbank)
    % RECOGNITION_PHASE_COLOR_5X5
    %   Fast vectorized color recognition
    
    I = im2double(I_rgb);
    [rows, cols, ~] = size(I);
    
    numK = numel(Hbank);
    Nmodels = size(MODEL.R, 1);
    
    % Compute feature maps
    BB_R = zeros(rows, cols, numK);
    BB_G = zeros(rows, cols, numK);
    BB_B = zeros(rows, cols, numK);
    
    Nkernel = ones(15) / (15 * 15);
    
    for k = 1:numK
        H = Hbank{k};
        
        Br = conv2(I(:,:,1), H, 'same');
        BB_R(:,:,k) = conv2(Br.^2, Nkernel, 'same');
        
        Bg = conv2(I(:,:,2), H, 'same');
        BB_G(:,:,k) = conv2(Bg.^2, Nkernel, 'same');
        
        Bb = conv2(I(:,:,3), H, 'same');
        BB_B(:,:,k) = conv2(Bb.^2, Nkernel, 'same');
    end
    
    % Vectorized classification
    BB_R_vec = reshape(BB_R, [], numK);
    BB_G_vec = reshape(BB_G, [], numK);
    BB_B_vec = reshape(BB_B, [], numK);
    
    distances = zeros(rows*cols, Nmodels);
    
    for n = 1:Nmodels
        dR = sum(abs(BB_R_vec - MODEL.R(n,:)), 2);
        dG = sum(abs(BB_G_vec - MODEL.G(n,:)), 2);
        dB = sum(abs(BB_B_vec - MODEL.B(n,:)), 2);
        
        distances(:, n) = (dR + dG + dB) / 3;
    end
    
    [~, labels] = min(distances, [], 2);
    ClassMap = reshape(labels, rows, cols);
end

function OUT = majority_voting(IN, w_dia)
    % MAJORITY_VOTING
    %   Block-wise majority voting
    
    [rows, columns] = size(IN);
    OUT = zeros(rows, columns);
    
    for i = 1:w_dia:rows
        i2 = min(i + w_dia - 1, rows);
        for j = 1:w_dia:columns
            j2 = min(j + w_dia - 1, columns);
            block = IN(i:i2, j:j2);
            OUT(i:i2, j:j2) = mode(block(:));
        end
    end
end

function stats = detect_and_visualize_cars(ClassMap, img_wallis, carClassIndices)
    % DETECT_AND_VISUALIZE_CARS
    %   Detects cars from segmentation, creates binary mask, and visualizes
    %
    %   stats = detect_and_visualize_cars(ClassMap, img_wallis, carClassIndices)
    %
    %   Inputs:
    %       ClassMap        : Segmented image (HxW with class labels)
    %       img_wallis      : Wallis-filtered RGB image (optional, for reference)
    %       carClassIndices : Array of class indices for cars, e.g. [4] or [4, 5]
    %
    %   Outputs:
    %       stats           : Structure array from regionprops with car info
    
    MIN_AREA_THRESHOLD = 20000;      
    MAX_AREA_THRESHOLD = 5000000;   
    MIN_SOLIDITY_THRESHOLD = 0.3;
    
    % =====================================================================
    % STEP 1: Create binary car mask (merge car classes)
    % =====================================================================
    fprintf('\nCreating binary car mask...\n');
    fprintf('Merging car classes: %s\n', mat2str(carClassIndices));
    
    car_mask = false(size(ClassMap));
    for carIdx = carClassIndices
        car_mask = car_mask | (ClassMap == carIdx);
    end
    
    fprintf('Initial car pixels: %d\n', sum(car_mask(:)));
    
    if sum(car_mask(:)) > 0
        fprintf('Applying morphological operations...\n');
        
        % Remove very small noise first (before filling)
        car_mask = bwareaopen(car_mask, 50);
        fprintf('  After bwareaopen (noise removal): %d pixels\n', sum(car_mask(:)));
        
        % Fill holes inside car regions
        car_mask = imfill(car_mask, 'holes');
        fprintf('  After imfill: %d pixels\n', sum(car_mask(:)));
        
        % Close small gaps (connect nearby car parts)
        car_mask = imclose(car_mask, strel('disk', 8));
        fprintf('  After imclose: %d pixels\n', sum(car_mask(:)));
        
        % Remove thin protrusions and noise
        car_mask = imopen(car_mask, strel('disk', 5));
        fprintf('  After imopen: %d pixels\n', sum(car_mask(:)));
    end
    
    % =====================================================================
    % STEP 3: Detect car blobs using regionprops
    % =====================================================================
    if sum(car_mask(:)) > 0
        stats = regionprops(car_mask, 'BoundingBox', 'Centroid', 'Area', 'Perimeter', 'PixelIdxList');
        
        fprintf('Found %d connected components\n', length(stats));
        
        % Filter by reasonable car size
        minCarArea = MIN_AREA_THRESHOLD;
        maxCarArea = MAX_AREA_THRESHOLD;
        validSize = [stats.Area] > minCarArea & [stats.Area] < maxCarArea;
        
        fprintf('Area range: [%d, %d] pixels\n', minCarArea, maxCarArea);
        fprintf('Detected blob areas: %s\n', mat2str(round([stats.Area])));
        
        % Filter by compactness (remove very elongated objects)
        compactness = ([stats.Perimeter].^2) ./ (4 * pi * [stats.Area]);
        validShape = compactness < 8;
        
        fprintf('Compactness values: %s\n', sprintf('%.2f ', compactness));
        
        % Combine filters
        validCars = validSize & validShape;
        
        fprintf('Valid size: %s\n', mat2str(find(validSize)));
        fprintf('Valid shape: %s\n', mat2str(find(validShape)));
        fprintf('Valid cars: %s\n', mat2str(find(validCars)));
        
        stats = stats(validCars);
        numCars = length(stats);
        
        fprintf('After filtering: %d valid cars\n', numCars);
        
        % ===== FINAL NOISE CLEANUP =====
        % Recreate mask with only valid cars
        final_mask = false(size(car_mask));
        for i = 1:numCars
            final_mask(stats(i).PixelIdxList) = true;
        end
        car_mask = final_mask;
        fprintf('After keeping only valid cars: %d pixels\n', sum(car_mask(:)));
    else
        stats = [];
        numCars = 0;
        fprintf('No car pixels detected\n');
    end
    
    % =====================================================================
    % STEP 4: Console output with car locations
    % =====================================================================
    fprintf('\n========== Car Detection Results ==========\n');
    fprintf('Total number of cars detected: %d\n\n', numCars);
    
    if numCars > 0
        fprintf('%-8s %-15s %-15s %-12s\n', 'Car #', 'X (pixels)', 'Y (pixels)', 'Area (px²)');
        fprintf('-------------------------------------------------------\n');
        
        for i = 1:numCars
            centroid = stats(i).Centroid;
            area = stats(i).Area;
            fprintf('%-8d %-15.1f %-15.1f %-12.0f\n', ...
                    i, centroid(1), centroid(2), area);
        end
    else
        fprintf('No cars detected in this image.\n');
    end
    fprintf('===========================================\n\n');
    
    % =====================================================================
    % FIGURE: Binary mask with bounding boxes
    % =====================================================================
    figure('Name','Car Detection on Binary Mask','NumberTitle','off');
    
    % Display binary mask (white = cars, black = background)
    imshow(car_mask);
    hold on;
    
    if numCars > 0
        % Draw bounding boxes and centroids
        for i = 1:numCars
            bb = stats(i).BoundingBox;  % [x, y, width, height]
            
            % bounding box
            rectangle('Position', bb, ...
                      'EdgeColor', 'blue', ...
                      'LineWidth', 2.5);
          
        end
        
        title(sprintf('Binary car mask: %d cars detected', numCars), ...
              'Color', 'white', 'FontSize', 14);
    else
        title('Binary car mask (no cars detected)', ...
              'Color', 'white', 'FontSize', 14);
    end
    
    hold off;
    fprintf('✓ Car detection figure displayed\n\n');
end
