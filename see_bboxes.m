load digitStruct.mat   %加载digitStructure矩阵，读取每张图片的结构信息
cnt = 0;
% 创建存储0-9十个数字的文件夹
for i = 1:10
    mkdir(['train_processed\',num2str(i-1)]);
end
% 与原始代码一样，不过多赘述，主要用于截取图像中的单个数字及标签
for i = 1:length(digitStruct)
    im = imread([digitStruct(i).name]);
    for j = 1:length(digitStruct(i).bbox)
        [height, width] = size(im);
        aa = max(digitStruct(i).bbox(j).top+1,1);
        bb = min(digitStruct(i).bbox(j).top+digitStruct(i).bbox(j).height, height);
        cc = max(digitStruct(i).bbox(j).left+1,1);
        dd = min(digitStruct(i).bbox(j).left+digitStruct(i).bbox(j).width, width);
        if digitStruct(i).bbox(j).label==10
            digitStruct(i).bbox(j).label = 0;
        end
        fprintf('%d %d %d %d %d %d %d\n',digitStruct(i).bbox(j).label ,i,j,aa,bb,cc,dd);
        % 将截取的单个数字图像按类保存到对应的文件夹下
        % 使用try catch是为了进行差错控制，防止截取出错时程序终止
        try
            imwrite(im(aa:bb, cc:dd, :),['train_processed\',num2str(digitStruct(i).bbox(j).label),'\',num2str(cnt),'.png']);
            cnt=cnt+1;
        catch
            fprintf('out range!');
        end 
        
    end
end