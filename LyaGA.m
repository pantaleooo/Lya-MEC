function [best_fitness_per_round, results, eh_real] = LyaGA(state, slot, NUM_DEVICE)
    % GA parameters
    POPULATION_SIZE = 20;
    MUTATION_RATE = 0.02;
    CROSSOVER_RATE = 0.6;
    MAX_GENERATIONS = 100;
   

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

    tau = parameter.tau;
    k = parameter.k;
    f_loc = parameter.f_loc;
    L = parameter.L;
    p_up = parameter.p_up;
    E_max = parameter.E_max;
    E_min = parameter.E_min;
    V = parameter.V;
    alpha = parameter.alpha;
    phi = parameter.phi;

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


    % Initial population
    population = randi([0, 2], POPULATION_SIZE, NUM_DEVICE);


    % Track the best solution of each round
    best_individual_per_round = zeros(MAX_GENERATIONS, NUM_DEVICE);
    best_fitness_per_round = zeros(1, MAX_GENERATIONS);

    for generation = 1:MAX_GENERATIONS

        % Calculate fitness
        fitness_scores = zeros(1, POPULATION_SIZE);
        new_population = zeros(size(population));
        for i = 1:POPULATION_SIZE
            results = struct();
            [results.obj1, results.obj2, results.total_Delay, results.total_Energy, results.device_Delay, results.device_Energy, results.fitness, results.new_individual, results.count_num,~] ...
                = fitness_function(population(i, :), NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter);
            fitness_scores(i) = results.fitness;
            new_population(i, :) = results.new_individual;
        end

        % Selection
        selected_population = selection(new_population, fitness_scores);
        

        % Record the best solution of this round
        [best_fitness, best_idx] = max(fitness_scores);
        best_individual_per_round(generation, :) = new_population(best_idx, :);

        best_fitness_per_round(generation) = best_fitness;

        % Crossover and mutation
        next_population = crossover(selected_population, CROSSOVER_RATE);
        next_population = mutation(next_population, MUTATION_RATE);

%         next_population = probSave(next_population, save_rate);

%         next_population(10,:) = best_individual_per_round(generation, :);

        population = next_population;

    end

    % Find the best individual in the final population
    [~, final_best_idx] = max(fitness_scores);
    best_individual = new_population(final_best_idx, :);
    [results.obj1, results.obj2, results.total_Delay, results.total_Energy, results.device_Delay, results.device_Energy, results.fitness, results.best_action, results.count_num, results.reliab_task] ...
        = fitness_function(best_individual, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter);
    results.Bat_level_virt = Bat_level_virt;
    results.execution_cost = results.obj2./V;
    results.best_obj = - results.fitness;
    results.eh_real = eh_real;
    % Optionally, plot the fitness evolution
%     if mod(slot, 100)==0
%         f = figure('Visible', 'off');
%         plot(best_fitness_per_round);
%         title('Best Fitness per Generation');
%         xlabel('Generation');
%         ylabel('Fitness');
%         filename = sprintf('Iteration_%d.png', slot);
%         saveas(f, filename);
%         close(f);
%     end
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






function alloc = task_assign(theta, theta_r, tau, M, F, N, L, lambda1, nu, location, r_up, r_down)
    ser_vul = zeros(4, M);  % 脆弱性、已分配资源(sqrt(频率) (Hz))、频率、原始编号
    task_vul = zeros(5, N); % 脆弱性、sqrt(工作量(cycles或Hz))、分配的服务器原始编号、原始编号

    ser_vul(1, :) = lambda1 ./ F;
    ser_vul(3, :) = F;
    ser_vul(4, :) = 1:M;

    [~, idx] = sort(ser_vul(1, :));
    ser_vul = ser_vul(:, idx);

    for i = 1:N
        task_vul(:, i) = [theta(i) * nu(i) * L(i); sqrt(theta(i) * L(i)); 0; i; 0];
    end
    task_vul(1, :) = theta .* nu .* L;
    task_vul(2, :) = sqrt(theta .* L);
    task_vul(4, :) = 1:N;
    [~, idx] = sort(task_vul(1, :), 'descend');
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





function [obj1, obj2, total_Delay, total_Energy, device_Delay, device_Energy, ftness, new_individual,count_num,reliab_task] = fitness_function(individual, NUM_DEVICE, bandwidth_up, bandwidth_down, compu_task_bit, download_task_bit, channel_gain, bat_level, Bat_level_virt, eh_real, parameter)
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

    task_ass = task_assign(compu_task_bit, download_task_bit, tau, M, F, NUM_DEVICE, L, lambda1, nu, individual, r_up, r_down);

    f_off = frequency_assign(compu_task_bit, M, F, NUM_DEVICE, L, task_ass);

    [obj1, obj2, total_Delay, total_Energy, device_Delay, device_Energy, new_individual, count_num] = obj_fun(Bat_level_virt, bat_level, E_min, E_max, eh_real, compu_task_bit, download_task_bit, NUM_DEVICE, V, L, k, p_up, p_down, individual, f_loc, f_off, r_up, r_down, task_ass, alpha, tau);
    ftness = - (obj1 + obj2); % The optimization problem is min, thus fitness is the negative of objective
    reliab_task = comput_sys_reliab(compu_task_bit, download_task_bit, NUM_DEVICE, F, nu, f_loc,beta, lambda1, lambda_loc, d, L,task_ass, f_off, new_individual, r_up, r_down);

    % Returning multiple outputs as per MATLAB syntax
end


function selected_population = selection(population, fitness)
    delta = 1.2;
    min_fit = delta * abs(min(fitness));
    fitness_virtual = fitness + min_fit;

    fitness_sum = sum(fitness_virtual);
    probabilities = fitness_virtual / fitness_sum;
    cum_probabilities = cumsum(probabilities);
    selected_population = zeros(size(population));

    for i = 1:size(population, 1)
        random_num = rand();
        index = find(cum_probabilities >= random_num, 1);
        selected_population(i, :) = population(index, :);
    end

    % Ensure the fittest individual is always selected
%     [~, index] = max(fitness_virtual);
%     selected_population(1, :) = population(index, :);
end



function offspring_population = crossover(selected_population, crossover_rate)
    offspring_population = zeros(size(selected_population));

    for i = 1:2:size(selected_population, 1)-1
        parent1 = selected_population(i, :);
        parent2 = selected_population(i+1, :);

        if rand() < crossover_rate
            % Single-point crossover
            crossover_point = randi([1, size(selected_population, 2)-1]);
            offspring1 = [parent1(1:crossover_point), parent2(crossover_point+1:end)];
            offspring2 = [parent2(1:crossover_point), parent1(crossover_point+1:end)];
        else
            % No crossover, parents are copied directly
            offspring1 = parent1;
            offspring2 = parent2;
        end

        offspring_population(i, :) = offspring1;
        offspring_population(i+1, :) = offspring2;
    end
end




function offspring_population = mutation(offspring_population, mutation_rate)
    mutation_mask = rand(size(offspring_population)) < mutation_rate;
    random_values = randi([0, 2], size(offspring_population));
    offspring_population(mutation_mask) = random_values(mutation_mask);
end



function offspring_population = probSave(offspring_population, save_rate)
    [idx, idy] = find(offspring_population == 2);
    % 遍历所有找到的位置
    for k = 1:length(idx)
        % 生成一个[0,1]区间的随机数
        randNum1 = rand();
        randNum2 = rand();
        if randNum1 <= save_rate
            % 根据概率 s 替换元素
            if randNum2 <= 0.5
                offspring_population(idx(k), idy(k)) = 1;
            else
                offspring_population(idx(k), idy(k)) = 0;
            end
        end
    end

end



