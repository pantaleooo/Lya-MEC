%%%%%%%%%%%%%%% 输入input包含 state, action, NUM_DEVICE, V
% %%%%%%%%%%%%% 输出results包含 obj1, obj2, device_Energy, best_obj, best_action, best_exe_cost, eh_real
   
function [results] = JTORA(input_para,relia_thres)

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
    xi = E_max_plus + V * phi ./ E_min;
    Bat_level_virt = bat_level - xi;

    eh_real = zeros(1,NUM_DEVICE);
    for i = 1:NUM_DEVICE
        if Bat_level_virt(i) <= 0
            eh_real(i) = ene_harv(i);
        else
            eh_real(i) = 0;
        end
    end

    % 初始化解,每一列只有一个元素能为1, 最后一行为1表示丢弃, 全为0表示本地计算
    decision = zeros(parameter.M+1, NUM_DEVICE);
    decision_record = [];
    reliab_record = [];
    bestFlightness_record = [];

    flag = 1;
    [~, ~, ~, ~, ~, ~, current_obj, ~, decision, ~, ~] ...
                = fitness_function(decision, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter);

    while flag==1
        [remove_flag,~,~,new_obj,new_decision,reliab] = exist_remove(current_obj,decision, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter);
        if remove_flag
            decision=new_decision;
            current_obj = new_obj;
            bestFlightness_record = [bestFlightness_record,current_obj];
            decision_record = cat(3,decision_record,decision);
            reliab_record = [reliab_record, reliab];
        else
            [exchange_flag,~,~,new_obj,new_decision,reliab] = exist_exchange(current_obj,decision, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter);
            if exchange_flag
                decision=new_decision;
                current_obj = new_obj;
                bestFlightness_record = [bestFlightness_record,current_obj];
                decision_record = cat(3,decision_record,decision);
                reliab_record = [reliab_record, reliab];
            end
        end
        if remove_flag==0 && exchange_flag==0
            flag=0;
        end
    end

    results = struct();

%     results.reliab_flag=0;
% 
%     for i=numel(reliab_record):-1:1
%         if reliab_record(i)>relia_thres
%             decision = decision_record(:,:,i);
%             results.reliab_flag=1;
%             break;
%         end
%     end    

    [results.obj1, results.obj2, ~, ~, ~, results.device_Energy, results.best_obj, results.best_action,results.decision, ~, results.reliab_task] ...
                = fitness_function(decision, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter);
    results.best_exec_cost = results.obj2/V;
    results.eh_real = eh_real;

end






% 卸载决策中遍历1，检查改为0是否更好
function [flag,index_server,index_device,new_obj,last_decision,reliab]=exist_remove(current_obj,decision, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter)
    reliab = 0;
    new_obj = 0;
    last_decision = decision;
    server_num = parameter.M;
    flag = 0;
    index_server=0;
    index_device=0;
    for i=1:server_num+1
        for j=1:NUM_DEVICE
            if decision(i,j)==1
                new_decision = decision;
                new_decision(i,j)=0;
                [~, ~, ~, ~, ~, ~, new_obj, ~, temp_decision, ~, reliab_temp] ...
                = fitness_function(new_decision, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter);
                reliab=prod(reliab_temp);
                if current_obj>0
                    tmp = 15/16;
                else
                    tmp = 16/15;
                end
                if new_obj<current_obj*tmp
                    flag = 1;
                    index_server = i;
                    index_device = j;
                    last_decision = temp_decision;
                    break;
                end
            end
        end
        if flag
            break; % 结束外层循环
        end
    end
end






function [flag,index_server,index_device,new_obj,last_decision,reliab] = exist_exchange(current_obj,decision, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter)
    reliab = 0;
    new_obj = 0;
    last_decision = decision;
    server_num = parameter.M;
    flag = 0;
    index_server=0;
    index_device=0;
    for i=1:server_num+1
        for j=1:NUM_DEVICE
            if decision(i,j)==0
                new_decision = decision;
                new_decision(:,j) = 0;
                new_decision(i,j) = 1;
                [~, ~, ~, ~, ~, ~, new_obj, ~, temp_decision, ~, reliab_temp] ...
                = fitness_function(new_decision, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter);
                reliab=prod(reliab_temp);
                if current_obj>0
                    tmp = 15/16;
                else
                    tmp = 16/15;
                end
                if new_obj < current_obj*tmp
                    flag = 1;
                    last_decision = temp_decision;
                    index_server = i;
                    index_device = j;
                    break;
                end
            end
        end
        if flag
            break; % 结束外层循环
        end
    end

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
            if tmp1 <= tau && tmp2 <= bat_level(i) && tmp2 <= E_max(i)
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





function [obj1, obj2, total_Delay, total_Energy, device_Delay, device_Energy, obj, new_individual, new_decision, count_num,reliab_task] = fitness_function(decision, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter)
    
    individual = sum(decision(1:end-1,:)); % 按列求和
    for i=1:NUM_DEVICE
        if decision(end,i)==1
            individual(i)=2;
        end
    end

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

    task_ass = decision(1:end-1,:);

    f_off = frequency_assign(compu_task_bit, M, F, NUM_DEVICE, L, task_ass);

    [obj1, obj2, total_Delay, total_Energy, device_Delay, device_Energy, new_individual, count_num] = obj_fun(Bat_level_virt, bat_level, E_min, E_max, eh_real, compu_task_bit, download_task_bit, NUM_DEVICE, V, L, k, p_up, p_down, individual, f_loc, f_off, r_up, r_down, task_ass, alpha, tau);

    obj = obj1 + obj2; % The optimization problem is min, thus fitness is the negative of objective
    
    % Returning multiple outputs as per MATLAB syntax

    new_decision = decision;
    for i=1:NUM_DEVICE
        if new_individual(i)==2 && individual(i)~=2
            new_decision(1:end-1,i)=0;
            new_decision(end,i)=1;
        end
    end
    
    task_ass = decision(1:end-1,:);
    f_off = frequency_assign(compu_task_bit, M, F, NUM_DEVICE, L, task_ass);
    [r_up, r_down] = trans_rate_compu(new_individual, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, p_up, p_edge, channel_gain, omega);
    reliab_task = comput_sys_reliab(compu_task_bit, download_task_bit, NUM_DEVICE, F, nu, f_loc,beta, lambda1, lambda_loc, d, L,task_ass, f_off, new_individual, r_up, r_down);

end






