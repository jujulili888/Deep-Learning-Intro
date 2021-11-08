load digitStruct.mat   %����digitStructure���󣬶�ȡÿ��ͼƬ�Ľṹ��Ϣ
cnt = 0;
% �����洢0-9ʮ�����ֵ��ļ���
for i = 1:10
    mkdir(['train_processed\',num2str(i-1)]);
end
% ��ԭʼ����һ����������׸������Ҫ���ڽ�ȡͼ���еĵ������ּ���ǩ
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
        % ����ȡ�ĵ�������ͼ���ౣ�浽��Ӧ���ļ�����
        % ʹ��try catch��Ϊ�˽��в����ƣ���ֹ��ȡ����ʱ������ֹ
        try
            imwrite(im(aa:bb, cc:dd, :),['train_processed\',num2str(digitStruct(i).bbox(j).label),'\',num2str(cnt),'.png']);
            cnt=cnt+1;
        catch
            fprintf('out range!');
        end 
        
    end
end