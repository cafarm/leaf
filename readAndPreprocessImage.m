function img = readAndPreprocessImage(filename)
    disp(['Processing: ' filename]);
    img = imread(filename);
    img = imresize(img, [227 227]);
end

