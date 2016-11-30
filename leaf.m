%% Load dataset table
t = readtable('leafsnap-dataset-images.txt', 'ReadVariableNames', true, 'Delimiter', 'tab');

%% Extract Histogram of Curvature over Scale feature
if ~exist('derived', 'dir')
    mkdir('derived');
end

objectsPerFile = 100;
nscales = 25;
nbins = 21;

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
if ~exist('tables', 'dir')
    mkdir('tables');
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
save('tables/complete-dataset', 'dataset');
