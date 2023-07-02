function [mammo_original, mammo_filtered] = dataset_filtered(path)
% Returns matlab matrix of the image, filtred with dwt2   
    im = imread(path);
    [cA1,cH1,cV1,cD1] = dwt2(im,'db2');
    mammo_original = im;
    mammo_filtered = cA1;
end