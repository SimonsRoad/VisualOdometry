function[result] = vel2pos(vel)
    [r,c] = size(vel);
    result = zeros(r,c);
    for i =2:1:r
        result(i,:) = result(i-1,:) + vel(i,:);
    end
end