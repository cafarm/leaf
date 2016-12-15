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

% This section uses the feature table created and extracted above to train a nearest neighbor model. The model is 
% actually trained and tested many times inorder to perform leave-one-out cross validation. The leave-one-out cross 
% validation approach entails looping through the set of field images only, removing 1 field image from the feature
% table, training the model with all lab and the remaining field images, querying the model with the removed field image
% and collecting the top 20 predictions for species classification.

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
    
    % Kumar et. al. found the histogram intersection distance to perform better than L1, L2, Bhattacharyya distance, 
    % and X2
    histIntersect = @(a,b)nscales - sum(min(a, b), 2);
    
    fprintf(['Running knnsearch on field image ' num2str(i) ' (id = ' num2str(fieldOnly.gnd{i,3}) ')...']);
    idx = knnsearch(fea, fieldOnly.fea(i,:), 'Distance', histIntersect, 'K', numel(gnd));
    
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

% This section plots the results of the nearest neighbor cross validation performed above. The accuracy is plotted
% against the number of top predictions considered. For instance, the model may be 70% accurate when you only consider
% if the true species actually matches the top prediction, but 90% accurate when you consider if the true species
% matches any of the top 5 predictions.

percentage = accuracy / nfieldImages * 100;
plot(percentage);

% The "ideal" is the results produced by Kumar et al. These were guesstimated by visually examining the chart in their
% paper. I am not aware of any actual published numbers.
hold on
ideal = [72 82 87 91 93 94.5 95.5 96 96.6 96.9 97.4 97.8 98.2 98.4 98.6 98.7 98.8 98.9 99 99.1];
plot(ideal);

title('Accuracy vs. Number of Top-Scoring Species Considered');
xlabel('Number of top-scoring species considered');
ylabel('Accuracy (percentage)');
legend('Nearest Neighbor (Cafaro)', 'Nearest Neighbor (Kumar et. al)');

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
cutoff = 25;
bar(1:numel(v(v>cutoff)), v(v>cutoff));
title('Species with Highest Error Rate');
set(gca, 'XTickLabel', k(v>cutoff), 'XTick', 1:numel(k(v>cutoff)));
set(gca, 'XTickLabelRotation', 60);
xlabel('Species');
ylabel('Error rate (percentage)');

%% Experiment with different starting scale to get most discrimination in fine scale with smooth vs serrated leaves

% This section attempts to find the ideal starting scale for feature extraction by examining a set of leaves that appear
% similar overall but where one is smooth and the other is serrated. The idea is to calculate the feature vector for
% these leaves by using different starting scales and then compare them against each other with the histogram
% intersection distance (the distance method used in the nearest neighbor model above). The starting scale that produces
% the greatest distance should be the starting scale that produces the greatest discrimination between these leaves in
% the model.

smooth1 = imread('dataset\segmented\field\diospyros_virginiana\12991999838005.png');
smooth2 = imread('dataset\segmented\field\diospyros_virginiana\12991999800320.png');
serrated = imread('dataset\segmented\field\amelanchier_arborea\13291782501779.png');

nscales = 25;
nbins = 21;
distanceSmooth = [];
distanceSerrated = [];
for i = 1:10
    hocs1 = calculateHocs(smooth1, nscales, nbins, i);
    hocs2 = calculateHocs(smooth2, nscales, nbins, i);
    hocs3 = calculateHocs(serrated, nscales, nbins, i);
    
    histIntersect = @(a,b)1 - sum(min(a, b), 2);
    
    distanceSmooth(end + 1) = histIntersect(hocs1(1:nbins), hocs2(1:nbins)); %#ok<SAGROW>
    distanceSerrated(end + 1) = histIntersect(hocs1(1:nbins), hocs3(1:nbins)); %#ok<SAGROW>
end

figure();
plot(distanceSmooth);
hold on;
plot(distanceSerrated);
title('Histogram Intersection Distance vs Starting Scale Radius');
xlabel('Starting scale radius');
ylabel('Intersection distance');
legend('Smooth vs. Smooth', 'Smooth vs. Serrated');

%% Experiment with different ending scale to get the most discrimination in coarse scale with non-lobed vs lobed leaves

% This section is similar to the one above but attempts to find the ideal ending scale by comparing non-lobed and lobed
% leaves.

nonlobed1 = imread('dataset\segmented\field\diospyros_virginiana\12991999838005.png');
nonlobed2 = imread('dataset\segmented\field\diospyros_virginiana\12991999800320.png');
lobed = imread('dataset\segmented\lab\quercus_shumardii\ny1156-01-3.png');

nscales = 25;
nbins = 21;
startRadius = 5;
endRadius = 40;
distanceNonlobed = [];
distanceLobed = [];
for i = startRadius:endRadius
    hocs1 = calculateHocs(nonlobed1, nscales, nbins, startRadius, i);
    hocs2 = calculateHocs(nonlobed2, nscales, nbins, startRadius, i);
    hocs3 = calculateHocs(lobed, nscales, nbins, startRadius, i);
    
    histIntersect = @(a,b)1 - sum(min(a, b), 2);
    
    distanceNonlobed(end + 1) = histIntersect(hocs1(end-nbins+1:end), hocs2(end-nbins+1:end)); %#ok<SAGROW>
    distanceLobed(end + 1) = histIntersect(hocs1(end-nbins+1:end), hocs3(end-nbins+1:end)); %#ok<SAGROW>
end

figure();
plot(startRadius:endRadius, distanceNonlobed);
hold on;
plot(startRadius:endRadius, distanceLobed);
title('Histogram Intersection Distance vs Ending Scale Radius');
xlabel('Ending scale radius');
ylabel('Intersection distance');
legend('Non-lobed vs. Non-lobed', 'Non-lobed vs. Lobed');

%% Extract features using convolutional neural networks

% This section uses a pre-trained convolution neural network to calculate feature vectors for each of the leaf images.

if ~exist('table', 'dir')
    mkdir('table');
end

labIdms = imageDatastore('dataset/images/lab', 'LabelSource', 'foldernames', 'IncludeSubfolders', true);
fieldIdms = imageDatastore('dataset/images/field', 'LabelSource', 'foldernames', 'IncludeSubfolders', true);

% Download and load pre-trained CNN network
cnnURL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-caffe-alex.mat';
cnnMatFile = fullfile(tempdir, 'imagenet-caffe-alex.mat');
if ~exist(cnnMatFile, 'file')    
    disp('Downloading pre-trained CNN model...');
    websave(cnnMatFile, cnnURL);
end
convnet = helperImportMatConvNet(cnnMatFile);

% Pre-processing for CNN
labIdms.ReadFcn = @(filename)readAndPreprocessImage(filename);
fieldIdms.ReadFcn = @(filename)readAndPreprocessImage(filename);

featureLayer = 'fc7';
labFeatures = activations(convnet, labIdms, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
fieldFeatures = activations(convnet, fieldIdms, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

%% Save extracted features to file

% This section saves the features vectors extracted from the section above to file.

l = cell(numel(labIdms.Labels), 1);
l(:) = {'lab'};
labGnd = [labIdms.Labels l];

l = cell(numel(fieldIdms.Labels), 1);
l(:) = {'field'};
fieldGnd = [fieldIdms.Labels l];

cnnDataset.gnd = [labGnd; fieldGnd];
cnnDataset.fea = [labFeatures fieldFeatures];
save('table/complete-cnn-dataset', 'cnnDataset');

disp(['Number of objects in cnn dataset: ' num2str(size(cnnDataset.gnd, 1))]);

%% Multiclass SVM

% This section uses a multiclass SVM classifier with the feature vectors extracted from the CNN above. Again the model
% is trained and tested multiple times. However, because SVM takes a while to train, more field images are left out at a
% time to reduce the overall runtime. This should only hurt, not help, accuracy. With more time, leave-one-out
% validation would be performed as it is in the nearest neighbor above.

file = load('table/complete-cnn-dataset');
cnnDataset = file.cnnDataset;

% Separate out the lab and field images
i = strcmp('lab', cellstr(cnnDataset.gnd(:,2)));
labOnly.gnd = cnnDataset.gnd(i,:);
labOnly.fea = cnnDataset.fea(:,i);
fieldOnly.gnd = cnnDataset.gnd(~i,:);
fieldOnly.fea = cnnDataset.fea(:,~i);

% Randomize field image order
[~, idx] = sort(rand(size(fieldOnly.gnd(:,1))));
fieldOnly.gnd = fieldOnly.gnd(idx,:);
fieldOnly.fea = fieldOnly.fea(:,idx);

nfieldImages = numel(fieldOnly.gnd(:,1));
nspecies = 20;
nper = 500;
missedSpecies = containers.Map();
accuracy = zeros(1, nspecies);
for i = 1:nper:nfieldImages
    disp(['Leaving out field images ' num2str(i) ' to ' num2str(i+nper) '.']);
    
    gnd = [labOnly.gnd(:,1); fieldOnly.gnd([1:i-1,i+nper:end],1)];
    fea = [labOnly.fea fieldOnly.fea(:,[1:i-1,i+nper:end])];
    
    svm = fitcecoc(fea, gnd(:,1), 'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
    
    for k = 1:nper
        if k+i>nfieldImages
            break;
        end
        
        [label, score] = predict(svm, fieldOnly.fea(:,k+i)');

        [~, idx] = sort(score, 'descend');

        speciesRank = svm.ClassNames(idx);

        rank = find(strcmp(cellstr(fieldOnly.gnd(k+i,1)), cellstr(speciesRank)), 1);
        if rank <= 20
            fprintf(['Matched after ' num2str(rank) ' top species considered\n']);
            accuracy(rank:end) = accuracy(rank:end) + 1;
        else
            s = char(label);
            fprintf(['NO MATCH: species = ' s ' \n']);
            if missedSpecies.isKey(s)
                missedSpecies(s) = missedSpecies(s) + 1;
            else
                missedSpecies(s) = 1;
            end
        end
    end
end

%% Plot SVM results

% This section plots the results of the nearest neighbor cross validation performed above. The accuracy is plotted
% against the number of top predictions considered, as it is with the nearest neighbor plot above.

percentage = accuracy / nfieldImages * 100;
plot(percentage);

% The "ideal" is the results produced by Kumar et al. These were guesstimated by visually examining the chart in their
% paper. I am not aware of any actual published numbers.
hold on
ideal = [72 82 87 91 93 94.5 95.5 96 96.6 96.9 97.4 97.8 98.2 98.4 98.6 98.7 98.8 98.9 99 99.1];
plot(ideal);

title('Accuracy vs. Number of Top-Scoring Species Considered');
xlabel('Number of top-scoring species considered');
ylabel('Accuracy (percentage)');
legend('Multiclass SVM (Cafaro)', 'Nearest Neighbor (Kumar et. al)');

errorRates = containers.Map();
species = missedSpecies.keys;
for i = 1:numel(species)
    count = sum(strcmp(species{i}, cellstr(cnnDataset.gnd(:,1))));
    nmissed = missedSpecies(species{i});
    errorRates(species{i}) = nmissed / count * 100;
end

figure();
k = cellstr(errorRates.keys);
v = errorRates.values;
v = [v{:}];
[v,i] = sort(v, 'descend');
k = k(i);
cutoff = 1.5;
bar(1:numel(v(v>cutoff)), v(v>cutoff));
title('Species with Highest Error Rate');
set(gca, 'XTickLabel', strrep(k(v>cutoff), '_', ' '), 'XTick', 1:numel(k(v>cutoff)));
set(gca, 'XTickLabelRotation', 60);
xlabel('Species');
ylabel('Error rate (percentage)');