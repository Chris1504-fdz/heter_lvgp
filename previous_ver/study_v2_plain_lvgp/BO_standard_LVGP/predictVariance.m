function varPred = predictVariance(Wnew, theta, d)
% Predict aleatoric variance using polynomial + exponential model

    % Build polynomial features
    Phi_new = ones(size(Wnew,1),1);
    for deg = 1:d
        Phi_new = [Phi_new, Wnew.^deg]; %#ok<AGROW>
    end

    % Optional: Add cross-terms (recommended for more expressive model)
    % Uncomment if desired:
    Phi_new = [Phi_new, Wnew(:,1).*Wnew(:,2), Wnew(:,1).*Wnew(:,3), Wnew(:,2).*Wnew(:,3)];

    % Compute log(sigma)
    log_sigma_new = Phi_new * theta;

    % Aleatoric variance = (k_al * exp(...))^2
    % Here k_al = 1 unless you include it separately
    varPred = exp(2 * log_sigma_new);
end