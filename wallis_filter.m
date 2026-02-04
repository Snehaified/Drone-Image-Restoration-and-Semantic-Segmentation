function enhanced = wallis_filter(img, window_size, target_mean, target_std, Amax, p)
    % WALLIS_FILTER - Applies Wallis filter for local contrast enhancement
    %
    % Inputs:
    %   img         - Input grayscale or RGB image (double, [0,1])
    %   window_size - Size of local window (e.g., 51 or 101)
    %   target_mean - Desired mean value (sigma_d in formula, e.g., 0.5)
    %   target_std  - Desired local contrast (sigma_d in formula, e.g., 0.2)
    %   Amax        - Maximum amplification factor (e.g., 1-5)
    %   p           - Weighting factor of mean compensation (e.g., 0.2)
    %
    % Output:
    %   enhanced    - Enhanced image with improved local contrast
    %
    % Wallis Formula:
    %   y(n1,n2) = [x(n1,n2) - x_bar(n1,n2)] * [Amax*sigma_d / (Amax*sigma_l(n1,n2) + sigma_d)]
    %              + [p*x_bar_d + (1-p)*x_bar(n1,n2)]
    %
    % where:
    %   x_bar(n1,n2) = local mean
    %   sigma_l(n1,n2) = local standard deviation (contrast)
    %   x_bar_d = desired mean (target_mean)
    %   sigma_d = desired contrast (target_std)
    
    % Get image dimensions
    [H, W, C] = size(img);
    
    % Initialize output
    enhanced = zeros(size(img));
    
    % Create averaging filter for local statistics
    h = ones(window_size, window_size) / (window_size^2);
    
    % Process each channel independently
    for c = 1:C
        channel = img(:,:,c);
        
        % Compute local mean: x_bar(n1,n2)
        x_bar = imfilter(channel, h, 'replicate');
        
        % Compute local standard deviation: sigma_l(n1,n2)
        % sigma_l = sqrt(E[(x - x_bar)^2])
        x_squared = channel.^2;
        x_bar_squared = imfilter(x_squared, h, 'replicate');
        variance_local = x_bar_squared - x_bar.^2;
        variance_local(variance_local < 0) = 0; % Handle numerical errors
        sigma_l = sqrt(variance_local);
        
        % Compute the gain factor according to the formula:
        % Amax * sigma_d / (Amax * sigma_l + sigma_d)
        gain = (Amax * target_std) ./ (Amax * sigma_l + target_std);
        
        % Compute the mean adjustment term:
        % p * x_bar_d + (1 - p) * x_bar(n1,n2)
        mean_term = p * target_mean + (1 - p) * x_bar;
        
        % Apply Wallis formula:
        % y = [x - x_bar] * gain + mean_term
        enhanced_channel = (channel - x_bar) .* gain + mean_term;
        
        % Clip to valid range [0, 1]
        enhanced_channel = max(0, min(1, enhanced_channel));
        
        enhanced(:,:,c) = enhanced_channel;
    end
end