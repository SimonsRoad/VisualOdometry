function[result] = getDiag(cell)
    for i =1:1:length(cell)
       result(i,:) = diag(cell{i});       
    end
end