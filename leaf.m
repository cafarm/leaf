%% Extract Histogram of Curvature over Scale feature

% This section extracts the Histogram of Curvature over Scale (HoCS) feature from each image as described by Kumar et. 
% al. in Leafsnap: A Computer Vision System for Automatic Plant Species Identification. The basic idea is to calculate
% the curvature around the parameter of each leaf at a number of different scales, bin these curvatures into histograms 
% for each scale, and then concatenate the resulting set of histograms into a feature vector for each leaf.
%
% This process takes a long time so we save a file every time we calculate a good number of feature vectors in order to
% avoid losing everything if something crashes. The resulting files are combine into a single file in the next section.

t = readtable('leafsnap-dataset-images.txt', 'ReadVariableNames', true, 'Delimiter', 'tab');

if ~exist('derived', 'dir')
    mkdir('derived');
end

% The number of scales to calculate the parameter curvature at
nscales = 25;

% The number of bins to use when creating a histogram of the curvatures for each scale
nbins = 21;

% The number of objects to save out to each file
objectsPerFile = 100;

for i = 1:ceil(height(t)/objectsPerFile)
    startIndex = (i - 1) * objectsPerFile + 1;
    endIndex = min(i * objectsPerFile, height(t));
    nobjects = endIndex - startIndex + 1;
    
    labels = cell(nobjects, 3);
    features = zeros(nobjects, nscales*nbins);
    
    for k = 1:nobjects
        objectIndex = startIndex + k - 1;
        
        img = imread(t.segmented_path{objectIndex});
        
        labels{k, 1} = t.species{objectIndex};
        labels{k, 2} = t.source{objectIndex};
        labels{k, 3} = t.file_id(objectIndex);
        
        fprintf(['Calculating feature for image ' num2str(objectIndex) '...']);
        try
            features(k,:) = calculateHocs(img, nscales, nbins);
            fprintf('Success\n');
        catch x
            fprintf(['FAILED: ' x.message '\n']);
        end
    end
    
    save(['derived/features_' num2str(startIndex) '-' num2str(endIndex)], 'features');
    save(['derived/labels_' num2str(startIndex) '-' num2str(endIndex)], 'labels');
    disp(['Saved features for objects ' num2str(startIndex) ' through ' num2str(endIndex)]);
end

%% Concatenate labels and features into a single file

% This section takes the set of feature tables created by the previous section, removes any objects that failed to
% compute a feature vector, and combines them all into a dataset suitable for machine learning.

if ~exist('table', 'dir')
    mkdir('table');
end
    
labelFiles = cellstr(ls('derived/labels*'));
featureFiles = cellstr(ls('derived/features*'));

assert(numel(featureFiles) == numel(labelFiles), 'Expected the same number of label and feature files');

allLabels = {};
allFeatures = [];

for i = 1:numel(labelFiles)
    l = load(['derived/' labelFiles{i}]);
    f = load(['derived/' featureFiles{i}]);
    
    % Remove objects that failed to compute feature vectors
    failed = all(f.features==0, 2);
    l.labels(failed,:) = [];
    f.features(failed,:) = [];
    
    allLabels = [allLabels; l.labels]; %#ok<AGROW>
    allFeatures = [allFeatures; f.features]; %#ok<AGROW>
end

dataset.gnd = allLabels;
dataset.fea = allFeatures;
save('table/complete-dataset', 'dataset');

disp(['Number of objects in dataset: ' num2str(size(dataset.fea, 1))]);

%% Nearest neighbor
file = load('table/complete-dataset');
dataset = file.dataset;

% Separate out the lab and field images
i = strcmp('lab', dataset.gnd(:,2));
labOnly.gnd = dataset.gnd(i,:);
labOnly.fea = dataset.fea(i,:);
fieldOnly.gnd = dataset.gnd(~i,:);
fieldOnly.fea = dataset.fea(~i,:);

for i = 1:2%
    
    % Kumar et. al. found the histogram intersection distance to perform better than L1, L2, Bhattacharyya distance, and X2
    histogramIntersectionDistance = @(a,b)nscales - sum(min(a, b), 2);

    knn = fitcknn([labOnly.fea; fieldOnly.fea([1:i-1,i+1:end],:)], ...
        [labOnly.gnd(:,1); fieldOnly.gnd([1:i-1,i+1:end],1)], ...
        'Distance', histogramIntersectionDistance, ...
        'NumNeighbors', 1);
    
    fprintf(['Predicting field image ' num2str(fieldOnly.gnd{i,3}) '...']);
    y = predict(knn, fieldOnly.fea(i,:));
    if strcmp(y{1}, fieldOnly.gnd{i,1})
        fprintf('Correct\n');
    else
        fprintf('WRONG\n');
    end
end
