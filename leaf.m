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
nscales = 25;

% Separate out the lab and field images
i = strcmp('lab', dataset.gnd(:,2));
labOnly.gnd = dataset.gnd(i,:);
labOnly.fea = dataset.fea(i,:);
fieldOnly.gnd = dataset.gnd(~i,:);
fieldOnly.fea = dataset.fea(~i,:);

nfieldImages = numel(fieldOnly.gnd(:,1));
nspecies = 20;
missedSpecies = containers.Map();
accuracy = zeros(1, nspecies);
for i = 1:nfieldImages
    gnd = [labOnly.gnd(:,1); fieldOnly.gnd([1:i-1,i+1:end],1)];
    fea = [labOnly.fea; fieldOnly.fea([1:i-1,i+1:end],:)];
    
    % Kumar et. al. found the histogram intersection distance to perform better than L1, L2, Bhattacharyya distance, and X2
    histogramIntersectionDistance = @(a,b)nscales - sum(min(a, b), 2);
    
    fprintf(['Running knnsearch on field image ' num2str(i) ' (id = ' num2str(fieldOnly.gnd{i,3}) ')...']);
    idx = knnsearch(fea, fieldOnly.fea(i,:), 'Distance', histogramIntersectionDistance, 'K', numel(gnd));
    
    speciesRank = {};
    for k = 1:numel(idx)
        species = gnd{idx(k),1};
        if any(strcmp(species, speciesRank))
            continue;
        end
        speciesRank{end + 1} = species; %#ok<SAGROW>
        
        if strcmp(species, fieldOnly.gnd{i,1})
            n = numel(speciesRank);
            fprintf(['Matched after ' num2str(n) ' nearest species checked\n']);
            accuracy(n:end) = accuracy(n:end) + 1;
            break;
        end
        
        if numel(speciesRank) >= nspecies
            s = fieldOnly.gnd{i,1};
            fprintf(['NO MATCH: species = ' s ' \n']);
            if missedSpecies.isKey(s)
                missedSpecies(s) = missedSpecies(s) + 1;
            else
                missedSpecies(s) = 1;
            end
            break;
        end
    end
end

%% Plot nearest neighbor results
percentage = accuracy / nfieldImages * 100;
plot(percentage);
hold on
ideal = [72 82 87 91 93 94.5 95.5 96 96.6 96.9 97.4 97.8 98.2 98.4 98.6 98.7 98.8 98.9 99 99.1];
plot(ideal);
title('Accuracy vs. Number of Nearest Species Considered');
xlabel('Number of nearest species considered');
ylabel('Accuracy (percentage)');
legend('Cafaro', 'Kumar et. al');

errorRates = containers.Map();
species = missedSpecies.keys;
for i = 1:numel(species)
    count = sum(strcmp(species{i}, dataset.gnd(:,1)));
    nmissed = missedSpecies(species{i});
    errorRates(species{i}) = nmissed / count * 100;
end

figure();
k = errorRates.keys;
v = errorRates.values;
v = [v{:}];
[v,i] = sort(v, 'descend');
k = k(i);
cutoff = 20;
bar(1:numel(v(v>cutoff)), v(v>cutoff));
title('Species with Highest Error Rate');
set(gca, 'XTickLabel', k(v>cutoff), 'XTick', 1:numel(k(v>cutoff)));
set(gca, 'XTickLabelRotation', 60);
xlabel('Species');
ylabel('Error rate (percentage)');
