% Initialization
time_steps = 1000;        % Number of time steps in the simulation
dt = 0.01;                % Time step size (smaller for higher resolution)
R = (0.01)^2;             % 1cm/s DVL standard deviation

M = [25, 25, 25, 10, 10, 10]';  % Mass-Inertia Matrix

% Initial Estimates
D_real = [0.35; 0.5; 0.35; 1; 0.4; 0.4]; % Real drag coefficients
mA_real = [15; 15; 10; 5; 8; 5];         % Real added masses

% Initial guess for EKF
D_k = [0.3; 0.45; 0.3; 0.9; 0.35; 0.35];  % Initial guess for drag coefficients
mA_k = [10; 12; 8; 4; 6; 4];              % Initial guess for added masses

% Thruster Allocation Matrix
T = [0.707  0.707   -0.707  -0.707  0   0   0   0;
     -0.707 0.707   -0.707  0.707   0   0   0   0;
     0  0   0   0   -1  1   1   -1;
     0.06   -0.06   0.06   -0.06  -0.218  -0.218  0.218  0.218;
     0.06   0.06   -0.06  -0.06   0.120  -0.120  0.120  -0.120;
    -0.188 0.188   0.188 -0.188  0   0   0   0]; % Thruster forces to DOF

% EKF Covariance matrices
P = diag([10; 10; 10; 10; 10; 10; 5; 5; 5; 5; 5; 5]); % Initial state covariance
Q = diag([0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1]); % Process noise covariance

% Simulation loop variables
v_measured = zeros(6, time_steps);  % Measured velocities over time
v_predicted = zeros(6, time_steps); % Predicted velocities over time
a_m = zeros(6, time_steps);         % Measured accelerations
v_measured(:,1) = 0;                % Initial velocity

% History storage for added mass, drag coefficients, and error history
mA_k_history = zeros(6, time_steps);  % History of added masses
D_k_history = zeros(6, time_steps);   % History of drag coefficients
mA_error_history = zeros(6, time_steps);  % Error history for added mass
D_error_history = zeros(6, time_steps);   % Error history for drag coefficients

for k = 2:time_steps
    % Sinusoidal thruster forces (input)
    tau_input = 15 * sin(2 * pi * k / time_steps) * ones(8,1);  % Sinusoidal excitation for all DOFs
    tau = T * tau_input;         % Compute forces applied by thrusters
    
    % Prediction Step
    a_p(:,k) = tau ./ (M + mA_k) - D_k .* v_measured(:, k-1)./(M + mA_k);  % Predicted acceleration
    v_predicted(:,k) = v_measured(:, k-1) + a_p(:,k) * dt;                % Predicted velocity

    % Simulated real dynamics (true model with noise)
    a_m(:,k) = tau ./ (M + mA_real) - D_real .* v_measured(:, k-1)./(M + mA_real);  % True acceleration
    v_measured(:, k) = v_measured(:, k-1) + a_m(:,k) * dt + randn(6,1) * sqrt(R);  % Update measured velocity with noise
    
    % EKF Prediction: State Prediction (mA_k and D_k update)
    X_pred = [mA_k; D_k];  % State vector [mA_k; D_k]
    
    % Linearize around the current state (Jacobian of the system dynamics)
    F = eye(12);  % Approximation of Jacobian for simplicity
    
    % Update covariance for the predicted state
    P_pred = F * P * F' + Q;
    
    % Measurement update
    H = [diag(v_measured(:,k-1)), diag(a_p(:,k))];  % Measurement model linearization
    
    % Innovation (residual)
    innovation = v_measured(:,k) - v_predicted(:,k);
    
    % Kalman Gain
    S = H * P_pred * H' + R * eye(6);  % Innovation covariance
    K = P_pred * H' / S;               % Kalman Gain
    
    % Update the state estimate
    X_update = X_pred + K * innovation;
    
    % Update covariance
    P = (eye(12) - K * H) * P_pred;
    
    % Split the updated state
    mA_k = X_update(1:6);
    D_k = X_update(7:12);
    
    % Store the updated values in history for plotting
    mA_k_history(:,k) = mA_k;  
    D_k_history(:,k) = D_k;
    
    % Compute the error between the real and estimated values
    mA_error = abs(mA_real - mA_k);  % Absolute error for added mass
    D_error = abs(D_real - D_k);     % Absolute error for drag coefficients
    
    % Store the error history for plotting
    mA_error_history(:,k) = mA_error;
    D_error_history(:,k) = D_error;
end

% Final estimation via Least Square Solution
X_lms = lsqminnorm(A, b);
mA_final = X_lms(1:6) - M;  % Final estimated added masses
D_final = X_lms(7:12);      % Final estimated drag coefficients

% Display final results
disp('Final Estimated Added Masses:');
disp(mA_final);
disp('Final Estimated Drag Coefficients:');
disp(D_final);

% Plot Vm (Measured Velocities) and Vp (Predicted Velocities)
figure;
for i = 1:6
    subplot(3, 2, i);
    plot(1:time_steps, v_measured(i,:), 'b', 'LineWidth', 1.5); hold on;
    plot(1:time_steps, v_predicted(i,:), 'r--', 'LineWidth', 1.5);
    title(['DOF ' num2str(i) ' Velocities']);
    xlabel('Time Steps');
    ylabel('Velocity');
    legend('Vm (Measured)', 'Vp (Predicted)');
end

% Plot Real vs Estimated Added Mass and Drag Coefficient
figure('Name', 'Real vs Estimated Added Mass');
for i = 1:6
    subplot(3, 2, i);
    % Plot Added Mass
    plot(1:time_steps, mA_k_history(i,:), 'r', 'LineWidth', 1.5); hold on;
    yline(mA_real(i), 'g--', 'LineWidth', 1.5);  % Plot real added mass
    title(['DOF ' num2str(i) ' Added Mass']);
    xlabel('Time Steps');
    ylabel('Added Mass');
    legend('Estimated', 'Real');
end

figure('Name', 'Real vs Estimated Drag Coefficients');
for i = 1:6
    subplot(3, 2, i);
    % Plot Drag Coefficients
    plot(1:time_steps, D_k_history(i,:), 'b', 'LineWidth', 1.5); hold on;
    yline(D_real(i), 'g--', 'LineWidth', 1.5);  % Plot real drag coefficient
    title(['DOF ' num2str(i) ' Drag Coefficient']);
    xlabel('Time Steps');
    ylabel('Drag Coefficient');
    legend('Estimated', 'Real');
end

% 3D Surface Plot for Added Mass and Drag Coefficients
figure('Name', '3D Surface Plots of Parameters');
subplot(1, 2, 1);
surf(1:time_steps, 1:6, mA_k_history);
title('Added Mass Evolution');
xlabel('Time Step');
ylabel('Degree of Freedom');
zlabel('Added Mass');
colorbar;

subplot(1, 2, 2);
surf(1:time_steps, 1:6, D_k_history);
title('Drag Coefficients Evolution');
xlabel('Time Step');
ylabel('Degree of Freedom');
zlabel('Drag Coefficient');
colorbar;

% Plot Error Convergence for Added Mass
figure('Name', 'Error Convergence for Added Mass');
for i = 1:6
    subplot(3, 2, i);
    plot(1:time_steps, mA_error_history(i,:), 'r', 'LineWidth', 1.5);
    title(['DOF ' num2str(i) ' Added Mass Error']);
    xlabel('Time Steps');
    ylabel('Error (Absolute)');
end

% Plot Error Convergence for Drag Coefficients
figure('Name', 'Error Convergence for Drag Coefficients');
for i = 1:6
    subplot(3, 2, i);
    plot(1:time_steps, D_error_history(i,:), 'b', 'LineWidth', 1.5);
    title(['DOF ' num2str(i) ' Drag Coefficient Error']);
    xlabel('Time Steps');
    ylabel('Error (Absolute)');
end
