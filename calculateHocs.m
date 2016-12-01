function hocs = calculateHocs(img, nscales, nbins, startScale, endScale, plotInterval)
    % Calculates the Histograms of Curvature over Scale feature for the given image as described by Kumar et. al. in 
    % Leafsnap: A Computer Vision System for Automatic Plant Species Identification.
    
    if nargin < 2
        nscales = 25;
    end
    if nargin < 3
        nbins = 21;
    end
    if nargin < 4
        startScale = 2;
    end
    if nargin < 5
        endScale = 26;
    end
    if nargin < 6
        plotInterval = [];
    end
    
    % Trace region boundaries
    boundaries = bwboundaries(img, 'noholes');
    if isempty(boundaries)
        error('Failed to trace boundary');
    end

    % Find the largest boundary (likely to be the leaf)
    [~, k] = max(cellfun(@(c)size(c,1), boundaries));
    boundary = boundaries{k};

    % Fill in boundary to remove any holes
    filled = poly2mask(boundary(:,2), boundary(:,1), size(img, 1), size(img, 2));

    % Find the region bounding boxes and filled area
    stats = regionprops(filled, 'BoundingBox', 'FilledArea');
    if isempty(stats)
        error('Failed to find region bounding box');
    end

    % Find the stats with the largest filled area (likely to be the leaf)
    [~, k] = max(arrayfun(@(s)s.FilledArea, stats));
    stats = stats(k);

    % Crop out the bounding box
    cropped = imcrop(filled, stats.BoundingBox);

    % Resize cropped image to common segmented area
    resized = imresize(cropped, sqrt(50000/stats.FilledArea));

    % Pad to max radius
    radii = round(linspace(startScale, endScale, nscales));
    resized = padarray(resized, [radii(end) radii(end)]);
    
    % Recalculate boundary from resized image
    boundaries = bwboundaries(resized, 'noholes');
    [~, k] = max(cellfun(@(c)size(c,1), boundaries));
    boundary = boundaries{k};

    % Calculate curvature histograms across each scale
    hocs = zeros(1, nscales * nbins);
    for i = 1:nscales
        radius = radii(i);

        % Calculate total pixel area of circle
        [xx, yy] = meshgrid(1:radius*2);
        circle = hypot(xx-radius, yy-radius) <= radius;
        circleArea = sum(circle(:));

        % For each boundary position, calculate the curvature
        npos = size(boundary, 1);
        curvatures = zeros(1, npos);
        for k = 1:npos
            xc = boundary(k, 2);
            yc = boundary(k, 1);
            
            % Isolate the section containing the circle
            r = radius;
            y1 = yc-r+1;
            y2 = yc+r;
            x1 = xc-r+1;
            x2 = xc+r;
            section = resized(y1:y2, x1:x2);
            
            % Calculate the area of the circle containing white pixels
            inside = circle & section;
            insideArea = sum(inside(:));
            
            curvatures(k) = insideArea / circleArea;

            % TODO: arclength curvature measure (this may not be important)
        end

        % TODO: Bilinear interpolation to do soft-binning of curvature values
        hocs((i-1)*nbins+1:i*nbins) = histcounts(curvatures, nbins) / npos;

        if ~isempty(plotInterval) && mod(i-1, plotInterval) == 0
            % Show image with curvatures
            figure();
            imshow(double(cat(3, resized, resized, resized)));
            hold on;
            scatter(boundary(:,2), boundary(:,1), 36, curvatures, 'filled');
            colormap('jet');
            text(5, size(resized, 1) - 10, ['radius = ' num2str(radius)], 'Color', 'white');
            %saveas(f, ['derived/image-' num2str(round(radius)) '.png']);

%             % Show curvature plot
%             figure();
%             x = 1:npos;
%             y = curvatures;
%             xx = [x;x];
%             yy = [y;y];
%             zz = zeros(size(xx));
%             surf(xx, yy, zz, yy, 'EdgeColor', 'interp', 'LineWidth', 5);
%             colormap('jet');
%             view(2);
%             title(['Plot of curvature measures, radius = ' num2str(radius)]);
%             %saveas(f, ['derived/plot-' num2str(round(radius)) '.png']);

            % Show histogram
            figure();
            histogram(curvatures, nbins);
            title(['Histogram of curvature measures, radius = ' num2str(radius)]);
            %saveas(f, ['derived/histogram-' num2str(round(radius)) '.png']);
        end
    end
end

