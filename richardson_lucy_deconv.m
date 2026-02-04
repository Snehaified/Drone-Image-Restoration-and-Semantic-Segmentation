function u_restored = richardson_lucy_deconv(d, P, num_iterations)
   
    % Get dimensions
    [H, W, num_channels_d] = size(d);
    [h_psf, w_psf, num_channels_P] = size(P);
    
    % Handle PSF channels: if grayscale PSF but color image, replicate PSF
    if num_channels_d == 3 && num_channels_P == 1
        % Replicate grayscale PSF to 3 channels
        P_rgb = repmat(P, [1, 1, 3]);
    elseif num_channels_d == 1 && num_channels_P == 3
        % Convert RGB PSF to grayscale if image is grayscale
        P_rgb = rgb2gray(P);
    else
        % Channels match or both are grayscale
        P_rgb = P;
    end
    
    % Stopping criteria parameters
    LOG_THRESHOLD = 0.15; % Threshold for LoG maximum intensity (adjust as needed)
    CHECK_INTERVAL = 5;   % Check stopping criteria every N iterations
    
    % Create Laplacian-of-Gaussian filter for sharpness measurement
    log_filter = fspecial('log', [9 9], 1.5);
    
    % Initialize output
    u_restored = zeros(size(d));
    
    % Process each channel
    for c = 1:num_channels_d
        % Extract channel
        d_channel = d(:,:,c);
        P_channel = P_rgb(:,:,min(c, size(P_rgb, 3)));
        
        % Normalize PSF
        P_channel = P_channel ./ (sum(P_channel(:)) + eps);
        
        % Pad PSF to match image size for FFT
        psf_padded = zeros(H, W);
        psf_padded(1:h_psf, 1:w_psf) = P_channel;
        % Circularly shift to center the PSF for proper FFT convolution
        psf_padded = circshift(psf_padded, -floor([h_psf, w_psf]/2));
        
        % Compute FFT of PSF and its conjugate (for P*)
        H_psf = fft2(psf_padded);
        H_psf_conj = conj(H_psf);
        
        % Initialize the estimate u_hat^(0)
        u_hat = d_channel;
        
        % Ensure non-negative values
        u_hat(u_hat < 0) = 0;
        d_channel(d_channel < 0) = 0;
        
        eps_val = 1e-6;
        converged = false;
        
        % Richardson-Lucy iterations with stopping criteria
        for t = 1:num_iterations
            % 1. Convolve current estimate with PSF (multiplication in freq domain)
            u_conv_P = real(ifft2(H_psf .* fft2(u_hat)));
            u_conv_P(u_conv_P < eps_val) = eps_val;
            
            % 2. Calculate ratio: observed / blurred estimate
            ratio = d_channel ./ u_conv_P;
            
            % 3. Calculate the Correction Term: Ratio CONVOLVED with P*
            % (using conjugate of H_psf for flipped PSF in frequency domain)
            correction_term = real(ifft2(H_psf_conj .* fft2(ratio)));
            
            % 4. Update u^(t+1) = u^(t) * Correction Term
            u_hat = u_hat .* correction_term;
            
            % Ensure the estimated image remains non-negative
            u_hat(u_hat < 0) = 0;
            
            % Check stopping criteria periodically
            if mod(t, CHECK_INTERVAL) == 0
                % Apply Laplacian-of-Gaussian filter to measure sharpness
                log_response = abs(imfilter(u_hat, log_filter, 'replicate'));
                max_log_intensity = max(log_response(:));
                
                % If sharpness exceeds threshold, we've converged
                if max_log_intensity > LOG_THRESHOLD
                    converged = true;
                    fprintf('Channel %d converged at iteration %d (LoG max: %.4f)\n', ...
                            c, t, max_log_intensity);
                    break;
                end
            end
        end
        
        % Report if max iterations reached without convergence
        if ~converged && t == num_iterations
            log_response = abs(imfilter(u_hat, log_filter, 'replicate'));
            max_log_intensity = max(log_response(:));
            fprintf('Channel %d reached max iterations %d (LoG max: %.4f)\n', ...
                    c, num_iterations, max_log_intensity);
        end
        
        % Store restored channel
        u_restored(:,:,c) = u_hat;
    end
end