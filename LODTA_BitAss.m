%%%%%%%%%%%%%%% 输入input包含 state, action, NUM_DEVICE, V
% %%%%%%%%%%%%% 输出results包含 obj1, obj2, device_Energy, best_obj, best_action, best_exe_cost, eh_real
   
function [results] = LODTA_BitAss(input_para)

%     NUM_DEVICE = 30;

    state = input_para.state;
    NUM_DEVICE = input_para.device_num;
    input_V = input_para.V;


    % 加载.mat文件
    struct = load('parameters.mat');
    parameter = struct.parameter;

    parameter.f_loc = parameter.f_loc(1:NUM_DEVICE);
    parameter.nu = parameter.nu(1:NUM_DEVICE);
    parameter.L = parameter.L(1:NUM_DEVICE);
    parameter.p_up = parameter.p_up(1:NUM_DEVICE);
    parameter.p_down = parameter.p_down(1:NUM_DEVICE);
    parameter.E_max = parameter.E_max(1:NUM_DEVICE);
    parameter.E_min = parameter.E_min(1:NUM_DEVICE);
    parameter.lambda_loc = parameter.lambda_loc(1:NUM_DEVICE);
    parameter.V = input_V;


    tau = parameter.tau;
    k = parameter.k;
    f_loc = parameter.f_loc;
    p_up = parameter.p_up;
    E_max = parameter.E_max;
    E_min = parameter.E_min;
    V = parameter.V;
    phi = parameter.phi;
    L = parameter.L;
    alpha = parameter.alpha;

    bat_level = state(1:5:end-2);
    ene_harv = state(2:5:end-2);
    compu_task_bit = state(3:5:end-2);
    download_task_bit = state(4:5:end-2);
    channel_gain = state(5:5:end);
    bandwidth_up = state(end-1);
    bandwidth_down = state(end);

    % Virtual battery energy level
    E_max_plus = min(max(k .* (f_loc .^ 3) * tau, p_up * tau), E_max);
    xi = E_max_plus + input_V * phi ./ E_min;
    Bat_level_virt = bat_level - xi;

    eh_real = zeros(1,NUM_DEVICE);
    for i = 1:NUM_DEVICE
        if Bat_level_virt(i) <= 0
            eh_real(i) = ene_harv(i);
        else
            eh_real(i) = 0;
        end
    end


    loc_obj = loc_obj_compu(Bat_level_virt, bat_level, E_max, eh_real, compu_task_bit, NUM_DEVICE, input_V, L, k, f_loc, alpha, tau);


    drop_obj = Bat_level_virt.*eh_real + input_V*alpha;

    location = zeros(1,NUM_DEVICE);
    
    for i=1:NUM_DEVICE
        tmp_loc = location;
        tmp_loc(i)=1;
        off_obj = off_obj_compu(i,tmp_loc, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, Bat_level_virt, eh_real, parameter, bat_level, tau, E_max);
        [~, min_index] = min([loc_obj(i),off_obj,drop_obj(i)]);
        location(i) = min_index-1;
    end

    results = struct();

    [results.obj1, results.obj2, ~, ~, ~, results.device_Energy, results.best_obj, results.best_action, ~,results.reliab_task] ...
                = fitness_function(location, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter);
    results.best_exec_cost = results.obj2/input_V;
    results.eh_real = eh_real;
    

end



function [r_up, r_down] = trans_rate_compu(location, B_up, B_down, theta, theta_r, p_up, p_edge, g, omega)
    offload_index = find(location == 1);
    offload_num = numel(offload_index);
    if ~isempty(offload_index)
        log_val_up = log2(1 + (p_up.*g) ./ omega);
        log_val_down = log2(1 + (p_edge.*g) ./ omega);

        r_up = B_up / offload_num .* log_val_up;
        r_down =B_down /offload_num .* log_val_down;
    else
        r_up = zeros(size(theta));
        r_down = zeros(size(theta_r));
    end
%     offload_index = find(location == 1);
%     if ~isempty(offload_index)
%         theta_sum = sum(theta(offload_index));
%         theta_r_sum = sum(theta_r(offload_index));
% 
%         log_val_up = log2(1 + (p_up.*g) ./ omega);
%         log_val_down = log2(1 + (p_edge.*g) ./ omega);
% 
%         r_up = theta / theta_sum .* B_up .* log_val_up;
%         r_down = theta_r / theta_r_sum .* B_down .* log_val_down;
%     else
%         r_up = zeros(size(theta));
%         r_down = zeros(size(theta));
%     end
end


function alloc = task_assign_random(theta, theta_r, tau, M, F, N, L, lambda1, nu, location, r_up, r_down)
    ser_vul = zeros(4, M);  % 脆弱性、已分配资源(sqrt(频率) (Hz))、频率、原始编号
    task_vul = zeros(5, N); % 脆弱性、sqrt(工作量(cycles或Hz))、分配的服务器原始编号、原始编号

    ser_vul(1, :) = lambda1 ./ F;
    ser_vul(3, :) = F;
    ser_vul(4, :) = 1:M;

    [~, idx] = sort(ser_vul(3, :), 'descend');
    ser_vul = ser_vul(:, idx);

    for i = 1:N
        task_vul(:, i) = [theta(i) * nu(i) * L(i); sqrt(theta(i) * L(i)); 0; i; 0];
    end
    task_vul(1, :) = theta .* nu .* L;
    task_vul(2, :) = sqrt(theta .* L);
    task_vul(4, :) = 1:N;
    [~, idx] = sort(task_vul(2, :), 'descend');
    task_vul = task_vul(:, idx);

    
    for i = 1:N
        inde = task_vul(4, i);
        if location(inde) == 1
            task_vul(5, i) = tau - theta(inde) / r_up(inde) - theta_r(inde) / r_down(inde);
        end
    end

    % Task to server assignment logic
    for i = 1:N
        inde = task_vul(4, i);
        if location(inde) == 1
            for m = 1:M
                if task_vul(2, i) * (ser_vul(2, m) + task_vul(2, i)) <= ser_vul(3, m) * task_vul(5, i)
                    tmp = 1;
                    for j = 1:N
                        if task_vul(3, j) == ser_vul(4, m) && task_vul(2, j) > task_vul(5, j) * ser_vul(3, m) / (ser_vul(2, m) + task_vul(2, i))
                            tmp = 0;
                        end
                    end
                    if tmp == 1
                        ser_vul(2, m) = ser_vul(2, m) + task_vul(2, i);
                        task_vul(3, i) = ser_vul(4, m);
                        break;
                    end
                end
            end
        end
    end

    % Compute server-task optimal allocation matrix
    alloc = zeros(M, N);
    for i = 1:N
        if task_vul(3, i) ~= 0
            alloc(task_vul(3, i), task_vul(4, i)) = 1;
        end
    end
end


function f_off = frequency_assign(theta, M, F, N, L, task_ass)
    % Initialize the f_off matrix with zeros
    f_off = zeros(M, N);

    precomputed_values = sqrt(L .* theta);
    
    % Assign frequencies based on task assignment and load
    for s = 1:M
        f_off(s, :) = precomputed_values .* task_ass(s, :);
    end
    

    for s = 1:M
        tmp = sum(f_off(s, :));
        for t = 1:N
            if task_ass(s, t) == 1 && tmp ~= 0
                f_off(s, t) = F(s) * f_off(s, t) / tmp;
            end
        end
    end
end



function loc_obj = loc_obj_compu(Bat_level_virt, bat_level, E_max, eh_real, theta, N, V, L, K, f_loc, alpha, tau)

    device_Delay = 0;
    device_Energy = 0;
    loc_obj = zeros(1,N);
    for i=1:N
        tmp1 = theta(i) * L(i) / f_loc(i); % Local computation delay
        tmp2 = K * f_loc(i)^2 * theta(i) * L(i); % Local computation energy
        if tmp1 <= tau && tmp2 <= bat_level(i) && tmp2 < E_max(i)
            device_Delay = tmp1;
            device_Energy = tmp2;
            loc_obj(i) = Bat_level_virt(i) * (eh_real(i) - device_Energy) + V * device_Delay;
        else
            device_Delay = alpha;
            device_Energy = 0;
            loc_obj(i) = 99999;
        end
        
    end

end




function off_obj = off_obj_compu(index,last_action, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, Bat_level_virt, eh_real, parameter, bat_level, tau, E_max)
    p_edge = parameter.p_edge;
    omega = parameter.omega;
    F = parameter.F;
    L = parameter.L;
    p_up = parameter.p_up;
    p_down = parameter.p_down;
    V = parameter.V;
    M = parameter.M;
    lambda1 = parameter.lambda;
    nu = parameter.nu;

    [r_up, r_down] = trans_rate_compu(last_action, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, p_up, p_edge, channel_gain, omega);

    task_ass = task_assign_random(compu_task_bit, download_task_bit, tau, M, F, NUM_DEVICE, L, lambda1, nu, last_action, r_up, r_down);

    f_off = frequency_assign(compu_task_bit, M, F, NUM_DEVICE, L, task_ass);

    if sum(f_off(:,index)) ~= 0
        device_Delay = compu_task_bit(index) / r_up(index) + compu_task_bit(index) * L(index) / sum(f_off(:,index)) + download_task_bit(index) / r_down(index); % Upload + Compute + Download delay
        device_Energy = p_up(index) * compu_task_bit(index) / r_up(index) + p_down(index) * download_task_bit(index) / r_down(index); % Upload + Download energy
        if device_Delay <= tau && device_Energy <= bat_level(index) && device_Energy <= E_max(index)
            off_obj = Bat_level_virt(index) * (eh_real(index) - device_Energy) + V * device_Delay;
        else
            off_obj = 99999;
        end
    else
        off_obj = 99999;
    end
   
end








function [obj1, obj2, total_Delay, total_Energy, device_Delay, device_Energy, new_location, count_num] = obj_fun(Bat_level_virt, bat_level, E_min, E_max, eh_real, theta, theta_r, N, V, L, K, p_up, p_down, location, f_loc, f_off, r_up, r_down, task_ass, alpha, tau)

    obj1 = 0; % Objective function term 1
    obj2 = 0; % Objective function term 2
    total_Energy = 0; % Total energy consumption for N devices
    total_Delay = 0; % Total delay for N devices
    device_Delay = zeros(1, N); % Delay for N devices
    device_Energy = zeros(1, N); % Energy consumption for N devices

    count1=0;
    count2=0;
    count3=0;
    count4=0;
    

    new_location = location;

    for i = 1:N
        if location(i) == 0
            % Local execution decision
            tmp1 = theta(i) * L(i) / f_loc(i); % Local computation delay
            tmp2 = K * f_loc(i)^2 * theta(i) * L(i); % Local computation energy
            if tmp1 <= tau && tmp2 <= bat_level(i) && tmp2 < E_max(i)
                device_Delay(i) = tmp1;
                device_Energy(i) = tmp2;
            else
                new_location(i) = 2;
                device_Delay(i) = alpha;
                device_Energy(i) = 0;
                count1 = count1 + 1;
            end
        elseif location(i) == 1
            % Offloading decision
            if sum(task_ass(:, i)) == 1 % Task assigned to a server
                tmp1 = theta(i) / r_up(i) + theta(i) * L(i) / sum(f_off(:, i)) + theta_r(i) / r_down(i); % Upload + Compute + Download delay
                tmp2 = p_up(i) * theta(i) / r_up(i) + p_down(i) * theta_r(i) / r_down(i); % Upload + Download energy
                if tmp1 <= tau && tmp2 <= bat_level(i) && tmp2 <= E_max(i)
                    device_Delay(i) = tmp1;
                    device_Energy(i) = tmp2;
                else
                    new_location(i) = 2;
                    device_Delay(i) = alpha;
                    device_Energy(i) = 0;
                    count2 = count2 + 1;
                end
            else % Server is overloaded, task is discarded
                new_location(i) = 2;
                device_Delay(i) = alpha;
                device_Energy(i) = 0;
                count3 = count3 + 1;
            end
        elseif location(i) == 2
            % Task discarded
            device_Delay(i) = alpha;
            device_Energy(i) = 0;
            count4 = count4 + 1;
        end
    end

    total_Delay = sum(device_Delay);
    total_Energy = sum(device_Energy);
    obj1 = sum(Bat_level_virt .* (eh_real - device_Energy));
    obj2 = sum(V .* device_Delay);
    count_num = [count1, count2, count3, count4];
end





function [obj1, obj2, total_Delay, total_Energy, device_Delay, device_Energy, obj, new_individual,count_num,reliab_task] = fitness_function(individual, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter)
    tau = parameter.tau;
    k = parameter.k;
    p_edge = parameter.p_edge;
    M = parameter.M;
    omega = parameter.omega;
    lambda1 = parameter.lambda;
    lambda_loc = parameter.lambda_loc;
    d = parameter.d;
    F = parameter.F;
    f_loc = parameter.f_loc;
    nu = parameter.nu;
    L = parameter.L;
    p_up = parameter.p_up;
    p_down = parameter.p_down;
    E_max = parameter.E_max;
    E_min = parameter.E_min;
    V = parameter.V;
    alpha = parameter.alpha;
    beta = parameter.beta;

    [r_up, r_down] = trans_rate_compu(individual, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, p_up, p_edge, channel_gain, omega);

    task_ass = task_assign_random(compu_task_bit, download_task_bit, tau, M, F, NUM_DEVICE, L, lambda1, nu, individual, r_up, r_down);

    f_off = frequency_assign(compu_task_bit, M, F, NUM_DEVICE, L, task_ass);

    [obj1, obj2, total_Delay, total_Energy, device_Delay, device_Energy, new_individual, count_num] = obj_fun(Bat_level_virt, bat_level, E_min, E_max, eh_real, compu_task_bit, download_task_bit, NUM_DEVICE, V, L, k, p_up, p_down, individual, f_loc, f_off, r_up, r_down, task_ass, alpha, tau);
    exe_cost = obj2 / V;
    obj = obj1 + obj2; % The optimization problem is min, thus fitness is the negative of objective

    reliab_task = comput_sys_reliab(compu_task_bit, download_task_bit, NUM_DEVICE, F, nu, f_loc,beta, lambda1, lambda_loc, d, L,task_ass, f_off, new_individual, r_up, r_down);
    % Returning multiple outputs as per MATLAB syntax
end