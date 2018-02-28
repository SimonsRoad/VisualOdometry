clc, clear, close all

for seq=0:1:10
    posP = importdata(strcat('Pred_Data/seq', int2str(seq), '_pospr.txt'))
    posA = importdata(strcat('Pred_Data/seq', int2str(seq), '_posgt.txt'))
    posStd = 1*sqrt(abs(importdata(strcat('Pred_Data/seq', int2str(seq), '_poscov.txt'))));

    fig = figure
    hold on
    plot(posA(:,1), posA(:,3), 'r-.')
    plot(posP(:,1), posP(:,3), 'b-.')
    grid on
    for i = 1:20:100%length(posA)
        index  = i;
        [x,y] = getEllipse(posP(index,1), posP(index,3), 3*posStd(index,1), 3*posStd(index,3));
        plot(posA(index,1), posA(index,3), 'r*', 'MarkerSize', 5)
        plot(x,y, 'b');
    end
    % plot(posA(1786,1), posA(1786,3), 'ro')
    % plot(posP(1786,1), posP(1786,3), 'bo')
    legend('Ground Truth', 'Prediction', 'Location', 'Best')
    saveas(fig, strcat('Images/seq', int2str(seq), '_cov.png'))

    % axis = 1
    % figure 
    % hold on
    % for i =1:5:length(posA)
    %     if mod(i,2) == 0
    %         plot([i, i], [posP(i,axis)  posP(i,axis)+posStd(i,1)], 'b')
    %         plot([i, i], [posP(i,axis)  posP(i,axis)-posStd(i,1)], 'b')
    %     else
    %         plot([i, i], [posP(i,axis)  posP(i,axis)+3*posStd(i,1)], 'g')
    %         plot([i, i], [posP(i,axis)  posP(i,axis)-3*posStd(i,1)], 'g')
    %     end
    % end
    % plot(posA(:,axis), 'r')
    % plot(posP(:,axis), 'b')
    % 
    % axis = 3
    % figure 
    % hold on
    % for i =1:5:length(posA)
    %     if mod(i,2) == 0
    %         plot([i, i], [posP(i,axis)  posP(i,axis)+posStd(i,1)], 'b')
    %         plot([i, i], [posP(i,axis)  posP(i,axis)-posStd(i,1)], 'b')
    %     else
    %         plot([i, i], [posP(i,axis)  posP(i,axis)+3*posStd(i,1)], 'g')
    %         plot([i, i], [posP(i,axis)  posP(i,axis)-3*posStd(i,1)], 'g')
    %     end
    % end
    % plot(posA(:,axis), 'r')
    % plot(posP(:,axis), 'b')
end