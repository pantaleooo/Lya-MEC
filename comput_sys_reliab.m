% task_ass, f_off都是M x N的数组
function reliab_task = comput_sys_reliab(theta, theta_r, N, F, nu, f_loc,beta, lambda,lambda_loc, d, L,task_ass, f_off, location, r_up, r_down)
    format long;
    reliab_task = zeros(1,N);
    for i=1:N
        if location(i)==0
            reliab_task(i)=exp(-lambda_loc(i)*nu(i)*theta(i)*L(i)/f_loc(i));
        elseif location(i)==1
            % 传输可靠性
            term1 = (theta(i)/r_up(i)+theta_r(i)/r_down(i))*beta;
            % edge server计算可靠性
            idx = find(task_ass(:,i)==1);
            term2 = lambda(idx)*exp(d*(F(idx)-f_off(idx,i))/F(idx));
            term3 = nu(i)*theta(i)*L(i)/f_off(idx,i);
%             fprintf("offload: %f\n",term2*term3);
            reliab_task(i) = exp(-term1-term2*term3);
        elseif location(i)==2
            reliab_task(i)=1;
        end
    end
    result = all(location == 2);
    if result
        reliab_task(:)=0;
    else
        idx = location==2;
        min_value = min(reliab_task);
        reliab_task(idx) = min_value;
    end

end


