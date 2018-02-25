clc, clear, close all

fName = 'mySeq01.txt';
data = importdata(fName);

gps = data(:,3:4);
quat = data(:,5:8);

lat = (gps(:,1) - gps(1,1))*10^6;
lon = (gps(:,2) - gps(1,2))*10^6;
plot(lon, lat, 'or')

for i = 1:1:length(quat)
    r(i,:) = reshape(quat2rotm(quat(i,:)), [1,9]);
end

newData = [r, lat, lon];


dlmwrite('myGoodSeq01.txt',newData,'delimiter',' ','precision',10)
