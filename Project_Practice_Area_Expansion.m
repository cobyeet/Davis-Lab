%[text] ## Advanced E/I Neuron Classification and Neurophysiological Analysis
%[text] 
%[text] for each cluster, make the cross/autocorrelelogram, plot the spike width, asymmetry index.
%[text] 
%[text] Author: Coby Bowman
%[text] Date: September 15, 2025
%[text] Dataset: Steinmetz et al., 2019 (VISp region)
%[text] Objective: This script performs a multi-stage analysis of neuronal data. It begins with a sophisticated classification of neurons into excitatory (E) and inhibitory (I) types, validates the classification, analyzes the suitability of this method across different cortical layers, and concludes with an analysis of the neurons' temporal dynamics and functional connectivity.
%[text] ## 1. Data Loading & Preprocessing
%[text] First, we load the necessary metadata and raw data, including channel locations, spike times, cluster assignments, and template waveforms. The primary goal is to isolate data from the **primary visual cortex (VISp)** for analysis.
%[text] ## 2. Enhanced E/I Classification Pipeline
%[text] This pipeline uses a robust, **multi-feature clustering approach** to discover natural E/I groupings in the data, avoiding the limitations of single-feature thresholding.
%[text] **2.1. Feature Extraction**
%[text] We extract two informative features from each neuron's average waveform:
%[text] - **Spike Width**: The full-width at half-minimum of the trough.
%[text] - **Peak-to-Trough Ratio**: The ratio of the repolarization peak to the hyperpolarization trough. \
%[text] **2.2. K-Means Clustering for Classification**
%[text] We use the **k-means clustering algorithm** to find the two most distinct groups in the 2D feature space. This is a more objective method than manually setting a threshold, especially for unimodal distributions. The cluster with the narrower average spike width is assigned the "putative inhibitory" label.
%[text] **2.3. Classification Visualization**
%[text] The results are visualized with two plots:
%[text] - **Scatter Plot**: Shows the two discovered clusters and their centers (centroids) in the 2D feature space.
%[text] - **Histogram**: Shows the spike width distributions for the two classes as determined by k-means. \
%[text] ## 3. Layer-Aware Bimodality Analysis
%[text] This section investigates *where* spike-width-based E/I classification is most effective by stratifying the analysis by cortical layer.
%[text] **3.1. Layer Assignment**
%[text] Neurons are grouped into putative cortical layers (e.g., L2/3, L4, L5, L6) based on the recording depth of their peak channel.
%[text] **3.2. Per-Layer Distribution Testing**
%[text] For each layer, we create spike width histograms and test if the distribution is unimodal (one group) or bimodal (two distinct E/I groups) using multiple statistical methods:
%[text] - **Hartigan's Dip Test**: A statistical test for multimodality. A low p-value suggests the presence of more than one mode.
%[text] - **Gaussian Mixture Model (GMM) Fitting**: We fit a two-component GMM to find the optimal separation between putative E and I populations.
%[text] - **Silhouette Analysis**: Measures how well-separated the GMM clusters are. Scores near 1 indicate a clear separation. \
%[text] **3.3. Layer-Specific Thresholds and Visualization**
%[text] The analysis identifies which layers have well-separated E/I populations (i.e., are bimodal). For these layers, a specific spike width threshold is calculated. The results are visualized with layer-stratified plots showing the histograms and GMM fits. This helps determine which cortical layers are suitable for this classification method.
%[text] ## 4. Statistical Validation via Permutation Testing
%[text] To determine if our global E/I ratio is biologically meaningful, we ask: *"Could this ratio have occurred by random chance?"* We use a **permutation test** to answer this.
%[text] **The Logic & Test**
%[text] - **Observed Ratio**: The E/I ratio calculated from our k-means classification.
%[text] - **Null Hypothesis**: Assumes the "E" and "I" labels are random and unrelated to neuronal properties.
%[text] - **Simulation**: We test this by repeatedly **shuffling** the E/I labels among the neurons and recalculating the ratio. This process builds a **null distribution** of ratios that could occur by chance. \
%[text] **The P-value**
%[text] The p-value is the proportion of outcomes in the null distribution that were as extreme or more extreme than our observed ratio. A **low p-value (p \< 0.05)** suggests our classification captured a real biological structure.
%[text] ## 5. Spike Timing Correlation Analysis
%[text] After classifying the neurons, we investigate their temporal relationships and potential functional connections using correlograms.
%[text] **5.1. Auto-Correlograms (ACGs)**
%[text] We calculate the auto-correlogram for each individual neuron. This is a histogram of time differences between a neuron's own spikes, which helps us assess:
%[text] - **Unit Isolation Quality**: A clear dip around time zero indicates a refractory period, confirming a well-isolated single unit.
%[text] - **Firing Patterns**: Peaks at short latencies can indicate burst firing. \
%[text] **5.2. Cross-Correlograms (CCGs)**
%[text] We calculate cross-correlograms for pairs of neurons. This is a histogram of time differences between spikes from two different neurons, allowing us to find:
%[text] - **Potential Monosynaptic Connections**: A sharp, narrow peak within a few milliseconds of time zero can indicate a direct synaptic connection (e.g., from an E cell to an I cell).
%[text] - **Shared Input**: A broader peak centered around zero can suggest two neurons are driven by a common, unobserved input. \
%[text] **5.3. Asymmetry Index for E-E Pairs**
%[text] For pairs of excitatory neurons, we calculate an **asymmetry index** based on the CCG. This metric quantifies the temporal ordering of firing.
%[text] - **Calculation**: (AUC\_post - AUC\_pre) / (AUC\_post + AUC\_pre), where AUC is the area under the curve in a small window just before (pre) and after (post) time zero.
%[text] - **Interpretation**: A positive value suggests a "feedforward" relationship (neuron 1 tends to fire before neuron 2), while a negative value suggests a "feedback" relationship. This helps map the flow of information within the local excitatory network. \
%[text] 
%[text] **Areas for Potential Improvement & Consideration**
%[text] While the script is excellent, here are a few suggestions that could make it even more robust:
%[text] 1. **Refining the Classification Algorithm:** \
%[text] - While `k-means` is a great choice, it assumes spherical clusters of similar variance. Your E and I populations might be better described by elliptical clusters. Consider using a **Gaussian Mixture Model (GMM)** not just for the bimodality test, but for the primary classification itself. GMMs can handle non-spherical clusters and provide probabilistic assignments, which can be useful for ambiguous neurons.
%[text] - Explore adding a third "unclassified" category for neurons that lie near the decision boundary, as not all cells can be confidently labeled E or I based on waveform alone. \
%[text] 1. **Correlogram Analysis Enhancements:** \
%[text] - **Baseline Correction:** For the CCGs, it's common practice to subtract a baseline "jitter" correlogram to control for slow, co-varying firing rates that aren't due to direct synaptic connections. This is often done by creating a shuffled CCG (jittering spike times within a small window) and subtracting it from the raw CCG to make true synaptic peaks stand out more clearly.
%[text] - **Significance Testing:** The current script identifies "example" connections by finding the maximum peak. A more rigorous approach would be to implement a statistical significance test on *every* CCG to automatically detect all putative connections that exceed a confidence threshold (often calculated from the baseline or a Poisson assumption). \
%[text] 1. **Layer Assignment:** \
%[text] - The current layer boundaries in `assignCorticalLayers` are hard-coded estimates. This is a reasonable starting point, but for a publication-quality analysis, these boundaries would ideally be confirmed histologically or by using current source density (CSD) analysis of the LFP to find the L4 sink. You correctly note this limitation in the function's comments. \
%[text] 1. **Code Robustness:** \
%[text] - The use of `try-catch` blocks in the feature extraction is excellent for handling problematic waveforms.
%[text] - The script relies on an external toolbox for Hartigan's Dip Test (`hartigansdipsigniftest`). It would be good practice to include a check at the beginning of the script to ensure this toolbox is in the MATLAB path, providing a more informative error message to the user if it's missing. You've correctly added a `try-catch` block as a fallback, which is a good solution. \
%%
%[text] ## ========================================================================
%[text] ## MAIN SCRIPT WORKFLOW
%[text] **1. Data Loading & Preprocessing**
%[text] First, we load the necessary metadata and raw data, including channel locations, spike times, cluster assignments, and template waveforms. The primary goal is to isolate data from the primary visual cortex (VISp) for analysis.
clear; clc; close all;

% Add path to npy-matlab library
addpath('/Users/cobybowman/Library/CloudStorage/Box-Box/Davis_Lab/npy-matlab-master/npy-matlab');
savepath;

%%
%[text] ## 1. Load channel + probe metadata

fprintf('Step 1: Loading channel and probe metadata...\n'); %[output:4bf00665]
brainloc = readtable('/Volumes/COBYB USB/Davis_Lab/Steinmetz_et_al_2019_9974357/nicklab/Subjects/Radnitz/2017-01-08/001/alf/channels.brainLocation.tsv', 'FileType', 'text', 'Delimiter', '\t');

%programmatically find all the unique brain areas recorded in your session.
% Find all unique brain regions in this recording session
allBrainAreas = brainloc.allen_ontology;
uniqueBrainRegions = unique(allBrainAreas);

% Optional: Clean up the list by removing non-specific labels like 'root' or 'void'
uniqueBrainRegions = uniqueBrainRegions(~strcmp(uniqueBrainRegions, 'root'));
uniqueBrainRegions = uniqueBrainRegions(~strcmp(uniqueBrainRegions, 'void'));

fprintf('Found %d unique brain regions to analyze.\n', length(uniqueBrainRegions)); %[output:2dccf135]
disp(uniqueBrainRegions); %[output:670b92fa]

probeIdx = readNPY('/Volumes/COBYB USB/Davis_Lab/Steinmetz_et_al_2019_9974357/nicklab/Subjects/Radnitz/2017-01-08/001/alf/channels.probe.npy');
rawRow   = readNPY('/Volumes/COBYB USB/Davis_Lab/Steinmetz_et_al_2019_9974357/nicklab/Subjects/Radnitz/2017-01-08/001/alf/channels.rawRow.npy');
probeFiles = readtable('/Volumes/COBYB USB/Davis_Lab/Steinmetz_et_al_2019_9974357/nicklab/Subjects/Radnitz/2017-01-08/001/alf/probes.rawFilename.tsv', 'FileType', 'text', 'Delimiter', '\t');
%[text] ## Main Analysis Loop and Results Structure

% Create a master struct to hold results from all regions
allRegionResults = struct();
% 6. Load pre-processed spike data
% The code processes neural spike data by identifying and filtering spikes that originate from clusters annotated as belonging to the VISp region. 
% It creates a structured dataset containing relevant spike times, cluster IDs, and waveforms for further analysis. 
fprintf('Step 6: Loading spike data...\n'); %[output:09e5d78f]
basePath = '/Volumes/COBYB USB/Davis_Lab/Steinmetz_et_al_2019_9974357/nicklab/Subjects/Radnitz/2017-01-08/001/alf/';
spikeTimes = readNPY([basePath 'spikes.times.npy']);
spikeClusters = readNPY([basePath 'spikes.clusters.npy']);
clusterPeakChan = readNPY([basePath 'clusters.peakChannel.npy']);
clusterAnnotation = readNPY([basePath 'clusters._phy_annotation.npy']);
templateWaveforms = readNPY([basePath 'clusters.templateWaveforms.npy']);

% <<< START OF THE NEW LOOP >>>
for i = 1:length(uniqueBrainRegions) %[output:group:00aef71c]
    currentRegion = uniqueBrainRegions{i};
    fprintf('\nProcessing region %d/%d: %s\n', i, length(uniqueBrainRegions), currentRegion); %[output:82297d39]
    fprintf('\n\n=======================================================\n'); %[output:36ad48d2]
    fprintf('Processing Brain Region: %s (%d of %d)\n', currentRegion, i, length(uniqueBrainRegions)); %[output:2d636619]
    fprintf('=======================================================\n'); %[output:21c8206a]

    % Step 7 (Modified): Filter for the CURRENT region's spikes
    fprintf('Step 7: Filtering for %s spikes...\n', currentRegion); %[output:03cec9da]

    goodClusters = find(clusterAnnotation >= 2);
    regionChannelMask = strcmp(brainloc.allen_ontology, currentRegion);
    regionChannels = find(regionChannelMask);

    if isempty(regionChannels)
    fprintf('Skipping %s: No channels found.\n', currentRegion);
       continue; % Skip to next region
    end

    regionClusterMask = ismember(clusterPeakChan + 1, regionChannels);
    goodRegionClusters = intersect(goodClusters, find(regionClusterMask));
    
    if length(goodRegionClusters) < 10 % Min neuron threshold
        fprintf('Skipping %s: Insufficient good clusters (%d).\n', currentRegion, length(goodRegionClusters));
     continue;
    end

    regionSpikeMask = ismember(spikeClusters, goodRegionClusters);

    % Create a temporary data struct for this region
    regionData = struct();
    regionData.name = currentRegion;
    regionData.clusters = goodRegionClusters;
    regionData.spikeTimes = spikeTimes(regionSpikeMask);
    regionData.spikeClusterIDs = spikeClusters(regionSpikeMask);
    regionData.waveforms = templateWaveforms(goodRegionClusters, :, :);
    regionData.nClusters = length(goodRegionClusters);
    regionData.nSpikes = sum(regionSpikeMask);
    fprintf('Created %s data struct with %d clusters and %d spikes.\n', currentRegion, regionData.nClusters, regionData.nSpikes); %[output:4bae02d6]
    
    % Call the validation function
    validateInputData(regionData); %[output:71a7f714]

    % RUN ALL ANALYSES on the current region's data
    % Note: Passing empty LFP data; this can be adapted later if needed.
    analysisResults = runEnhancedEIClassification(regionData, [], []); %[output:19cb6f84] %[output:5a0f0243] %[output:7e23ba33] %[output:48b04b78] %[output:213832c2] %[output:2a2be6f3] %[output:3fe4c203]


    layerAwareResults = runLayerAwareEIAnalysis(regionData, analysisResults.waveformFeatures, clusterPeakChan, rawRow); %[output:345a75e1] %[output:2f9fe308] %[output:2b69fb5d] %[output:3969cd2f] %[output:442532e7]
    analysisResults.layerAwareAnalysis = layerAwareResults;
    
    validationResults = validateEIClassification(regionData, analysisResults.enhancedClassification, brainloc, clusterPeakChan, currentRegion, 5000); %[output:2d86a759] %[output:5c8b8866]
    analysisResults.permutationTest = validationResults;
    
    correlationAnalysis = runSpikeCorrelationAnalysis(regionData, analysisResults, clusterPeakChan, rawRow); %[output:7eca2388] %[output:4f8ca4a3] %[output:6271552f]
    analysisResults.correlationAnalysis = correlationAnalysis;
    
    % --- NEW CODE TO STORE RESULTS ---
    % Sanitize the region name to use as a struct field name
    cleanRegionName = strrep(currentRegion, '/', '_'); 

    % Store all results in the master struct
    allRegionResults.(cleanRegionName) = analysisResults;
    allRegionResults.(cleanRegionName).rawData = regionData;

 end % <<< END OF THE NEW LOOP >>> %[output:group:00aef71c]

    fprintf('\n\n=== All brain regions processed! ===\n');
%%
%[text] ## \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
%[text] ## HELPER FUNCTIONS (All function definitions must be at the end of the script)
%[text] ## \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
%[text] ##  --- LFP Loading Function ---
function data = load_lfp_rows(filename, nChannels, rowIdx)
    % filename : full path to .lf.bin
    % nChannels: number of recorded channels
    % rowIdx   : which rows you want (e.g. VISp rows)
    % Returns: matrix [length(rowIdx) x nTimepoints]
    fid = fopen(filename, 'r');
    raw = fread(fid, [nChannels, Inf], 'int16=>double'); % load all channels
    fclose(fid);
    data = raw(rowIdx+1, :);  % MATLAB is 1-indexed, rawRow is 0-indexed
end


%%
%[text] ## --- Step 9: Enhanced Excitatory/Inhibitory Classification System ---
%[text] The runEnhancedEIClassification function orchestrates a series of steps to classify neural waveform data 
%[text]  into excitatory and inhibitory categories. It begins by extracting waveform features from the provided data,
%[text]  then analyzes the distribution characteristics of these features. Based on this analysis, it determines 
%[text]  the best classification strategy to apply. The function then executes the classification, validates the 
%[text]  results, and generates visualizations to illustrate the findings. Finally, it compiles all results into a 
%[text]  structured output for further analysis or reporting.

function enhancedResults = runEnhancedEIClassification(v1Data, vispLFP, timeVector)
    % Main function to run the enhanced E/I classification
    fprintf('\n\n=== Enhanced E/I Classification Analysis ===\n');
    
    % Extract waveform features
    %   The function calls extractComprehensiveWaveformFeatures to obtain relevant features from the input data,
    %   which are essential for classification.
    fprintf('Extracting comprehensive waveform features...\n');
    waveformFeatures = extractComprehensiveWaveformFeatures(v1Data);
    
    % Detect distribution modality
    %   This step analyzes the characteristics of the extracted features to understand their distribution, 
    %   which is crucial for determining the classification approach.
    fprintf('Analyzing distribution characteristics...\n');
    distributionAnalysis = analyzeDistributionModality(waveformFeatures);
    
    % Determine optimal classification strategy
    %   Based on the waveform features and their distribution, this line determines the optimal strategy 
    %   for classification.
    fprintf('Determining optimal classification strategy...\n');
    classificationStrategy = determineClassificationStrategy(waveformFeatures, distributionAnalysis);
    
    % Apply enhanced classification
    %   The function applies the chosen classification strategy to the waveform features to classify them as excitatory or inhibitory.
    fprintf('Applying enhanced E/I classification...\n');
    enhancedClassification = applyEnhancedClassification(waveformFeatures, classificationStrategy);
    
    % Validate results (using placeholder)
    fprintf('Validating classification accuracy...\n');
    validationResults = validateClassificationAccuracy(enhancedClassification, waveformFeatures);
    
    % Generate comprehensive plots
    fprintf('Generating visualization plots...\n');
    visualizationFigures = generateEnhancedClassificationPlots(waveformFeatures, enhancedClassification);
    
    % Compile results
    %   Here, all relevant results are compiled into a structured output, making it easy to access and review the findings.
    enhancedResults = struct();
    enhancedResults.waveformFeatures = waveformFeatures;
    enhancedResults.distributionAnalysis = distributionAnalysis;
    enhancedResults.classificationStrategy = classificationStrategy;
    enhancedResults.enhancedClassification = enhancedClassification;
    enhancedResults.validationResults = validationResults;
    enhancedResults.visualizationFigures = visualizationFigures;
    enhancedResults.summary = generateClassificationSummary(enhancedClassification, validationResults);
    
    fprintf('Enhanced E/I classification complete!\n');
    printSummaryResults(enhancedResults.summary);
end


    %   The extractComprehensiveWaveformFeatures function processes neural waveform data to compute a range of features
    %   that are useful for analyzing and classifying neural spikes. It initializes arrays to store features for each cluster of waveforms. The function iterates through 
    %   each cluster, calculates the desired features, and handles any errors that may arise during processing. 
    %   The results are stored in a structured format, allowing for easy access and further analysis.
function features = extractComprehensiveWaveformFeatures(v1Data)
    % Extract comprehensive waveform features for classification
    nClusters = v1Data.nClusters;
    features = struct();
    
    % Initialize feature arrays
    features.spikeWidth = zeros(nClusters, 1);
    features.peakToTroughRatio = zeros(nClusters, 1);
    features.peakToTroughLatency = zeros(nClusters, 1);
    features.troughTopeak = zeros(nClusters, 1);
    features.amplitude = zeros(nClusters, 1);
    features.asymmetryIndex = zeros(nClusters, 1);
    features.repolarizationSlope = zeros(nClusters, 1);
    features.afterHyperpolarization = zeros(nClusters, 1);
    features.waveformEnergy = zeros(nClusters, 1);
    features.validNeurons = false(nClusters, 1);
    features.clusterIDs = v1Data.clusters;
    
    % Sampling parameters
    %   The sampling rate is set, and the time step is calculated to convert the sampling frequency into milliseconds, 
    %   which will be used in feature calculations.

    samplingRate = 30000; % AP band sampling rate (Hz)
    %   If the original Local Field Potential (LFP) recording was sampled at 2,500 Hz, using a higher
    %   sampling rate, such as 30,000 Hz, may be necessary for specific analyses, 
    %   particularly when examining high-frequency components like action potentials.
    %   The higher sampling rate allows for better resolution of rapid changes in the waveform, which is essential for extracting features such as spike width and peak-to-trough ratios.
    timeStep = 1/samplingRate * 1000; % Convert to ms
    
    % The function iterates over each cluster of waveforms. 
    %   It uses a try-catch block to handle potential errors gracefully. 
    %   The waveform for the current cluster is extracted.
    %   Feature Calculations: Inside the loop, various features are calculated:
    %       Spike Width: Calculated as the full width at half maximum (FWHM).
    %       Peak-to-Trough Ratio: The ratio of peak value to the absolute trough value.
    %       Latencies and Amplitude: Calculated based on indices of the peak and trough.
    %       Asymmetry Index: A custom function is called to compute this feature.
    %       Repolarization Slope and After Hyperpolarization: These are derived from the waveform characteristics.
    %       Waveform Energy: The sum of squares of the waveform values.
    for i = 1:nClusters
        try
            waveform = squeeze(v1Data.waveforms(i, :, 1));
            [troughVal, troughIdx] = min(waveform);
            [peakVal, peakIdx] = max(waveform);
            
            if isnan(troughVal) || isnan(peakVal) || troughVal >= peakVal
                continue;
            end
            
            features.validNeurons(i) = true;
            
            % Spike Width (FWHM)
            halfTrough = troughVal / 2;
            belowHalf = find(waveform <= halfTrough);
            if length(belowHalf) >= 2
                features.spikeWidth(i) = (belowHalf(end) - belowHalf(1)) * timeStep;
            else
                features.spikeWidth(i) = NaN;
            end
            
            % Peak-to-Trough Ratio
            if abs(troughVal) > eps
                features.peakToTroughRatio(i) = peakVal / abs(troughVal);
            else
                features.peakToTroughRatio(i) = NaN;
            end
            
            features.peakToTroughLatency(i) = abs(peakIdx - troughIdx) * timeStep;
            features.troughTopeak(i) = (peakIdx - troughIdx) * timeStep;
            features.amplitude(i) = peakVal - troughVal;
            features.asymmetryIndex(i) = calculateAsymmetryIndex(waveform, troughIdx);
            
            if peakIdx > troughIdx
                features.repolarizationSlope(i) = (peakVal - troughVal) / ((peakIdx - troughIdx) * timeStep);
            else
                features.repolarizationSlope(i) = NaN;
            end
            
            if peakIdx < length(waveform)
                postPeakMin = min(waveform(peakIdx:end));
                features.afterHyperpolarization(i) = peakVal - postPeakMin;
            else
                features.afterHyperpolarization(i) = NaN;
            end
            
            features.waveformEnergy(i) = sum(waveform.^2);
            
        catch ME
            fprintf('Warning: Error processing cluster %d: %s\n', i, ME.message);
            continue;
        end
    end
    fprintf('Extracted features from %d/%d valid neurons\n', sum(features.validNeurons), nClusters);
end

    %   The analyzeDistributionModality function evaluates the distribution characteristics of neuron 
    %   features by checking for valid data points and performing statistical tests to determine if 
    %   the distributions are unimodal or bimodal. If there are insufficient valid neurons, it warns 
    %   the user and suggests a fallback strategy. The function then tests the bimodality of the spike 
    %   width and peak-to-trough ratio, analyzes combined features, and recommends a strategy based on the results. 
    %   Finally, it prints the analysis results, including modality types and p-values.
function distAnalysis = analyzeDistributionModality(features)
    distAnalysis = struct();
    % Here, validIdx is created to identify valid neurons by checking three conditions: 
    % the neuron must be valid, the spike width must not be NaN, and it must be greater than zero.
    validIdx = features.validNeurons & ~isnan(features.spikeWidth) & features.spikeWidth > 0;
  
    % If there are fewer than 10 valid neurons, a warning is printed,
    % and the function sets isValid to false, assigns a fallback strategy, and exits early.
    if sum(validIdx) < 10
        fprintf('Warning: Insufficient valid neurons for distribution analysis\n');
        distAnalysis.isValid = false;
        distAnalysis.recommendedStrategy = 'default_literature'; % Fallback strategy
        return;
    end
    % If there are enough valid neurons, the function sets isValid to true and extracts
    % the spike widths and peak-to-trough ratios corresponding to valid neurons.
    distAnalysis.isValid = true;
    spikeWidths = features.spikeWidth(validIdx);
    peakTroughRatios = features.peakToTroughRatio(validIdx);
    
    distAnalysis.spikeWidth = testBimodality(spikeWidths, 'Spike Width');
    distAnalysis.peakTroughRatio = testBimodality(peakTroughRatios, 'Peak-to-Trough Ratio');
    distAnalysis.combinedFeatures = analyzeCombinedFeatures(spikeWidths, peakTroughRatios);
    distAnalysis.recommendedStrategy = determineDistributionStrategy(distAnalysis);
    
    fprintf('Distribution Analysis Results:\n');
    fprintf('  Spike Width: %s (p=%.4f)\n', distAnalysis.spikeWidth.modalityType, distAnalysis.spikeWidth.pValue);
    fprintf('  Peak-Trough Ratio: %s (p=%.4f)\n', distAnalysis.peakTroughRatio.modalityType, distAnalysis.peakTroughRatio.pValue);
    fprintf('  Recommended Strategy: %s\n', distAnalysis.recommendedStrategy);
end

function classificationStrategy = determineClassificationStrategy(~, ~)
    % Determine the optimal classification strategy.
    % For this pipeline, we will use a more robust 2D k-means clustering approach.
    
    fprintf('Selecting k-means clustering strategy with 2 features...\n');
    classificationStrategy = struct(); 
    %   The method for classification is set to 'kmeans_2D', 
    %   and the features used for clustering are specified as a cell array containing spikeWidth and peakToTroughRatio. 
    %   This structure allows for easy access to the classification method and the features later in the code.
    classificationStrategy.method = 'kmeans_2D';
    classificationStrategy.features = {'spikeWidth', 'peakToTroughRatio'};
end

    %   The applyEnhancedClassification function takes neuron feature data and a classification strategy as inputs. 
    %   It uses GMM clustering to categorize neurons into excitatory and inhibitory groups based on their spike width and peak-to-trough ratio. 
    function classification = applyEnhancedClassification(features, strategy)
    % Applies a 2D Gaussian Mixture Model (GMM) to classify neurons.
    
    classification = struct();
    nNeurons = length(features.validNeurons);
    
    % Initialize outputs
    classification.isExcitatory = false(nNeurons, 1);
    classification.isInhibitory = false(nNeurons, 1);
    classification.isUnclassified = true(nNeurons, 1);
    classification.strategy = strategy; 

    % --- GMM Clustering ---
    % 1. Prepare the data for clustering from valid neurons
    validIdx = features.validNeurons & ~isnan(features.spikeWidth) & ~isnan(features.peakToTroughRatio);
    featureData = [features.spikeWidth(validIdx), features.peakToTroughRatio(validIdx)];
    
    if size(featureData, 1) < 10 % GMM needs a few points to converge
        fprintf('Warning: Not enough valid neurons for GMM clustering.\n');
        return;
    end
    
    % 2. Run GMM to find 2 clusters (putative E and I)
    fprintf('Running Gaussian Mixture Model (GMM) clustering on %d neurons...\n', size(featureData, 1));
    options = statset('MaxIter', 1000);
    
    % --- FIX IS HERE ---
    % Added 'RegularizationValue' to prevent ill-conditioned covariance error
    try
        gmm_model = fitgmdist(featureData, 2, 'Options', options, 'RegularizationValue', 1e-5);
    catch ME
        fprintf('Warning: GMM clustering failed for this region: %s\n', ME.message);
        % As a fallback, you could use kmeans here if you wanted
        % For now, we will just return and leave neurons as unclassified
        return;
    end
    
    % 3. Get the posterior probabilities for each neuron belonging to each cluster
    posterior_probs = posterior(gmm_model, featureData);
    
    % 4. Identify which cluster is inhibitory (has the narrower mean spike width)
    [~, inhClusterNum] = min(gmm_model.mu(:, 1)); % Find index of the mean with smaller spike width
    
    % 5. Create final classification masks based on which probability is higher
    validIndices = find(validIdx);
    for i = 1:length(validIndices)
        neuronIdx = validIndices(i);
        % Assign neuron to the cluster for which it has the highest posterior probability
        [~, assignedCluster] = max(posterior_probs(i, :));
        if assignedCluster == inhClusterNum
            classification.isInhibitory(neuronIdx) = true;
        else
            classification.isExcitatory(neuronIdx) = true;
        end
    end
    classification.isUnclassified(validIdx) = false;
    classification.gmm_model = gmm_model; % Save the model for inspection
    
    % 6. Generate Summary Stats
    summary = struct();
    summary.nExcitatory = sum(classification.isExcitatory);
    summary.nInhibitory = sum(classification.isInhibitory);
    nValid = summary.nExcitatory + summary.nInhibitory;
    
    if nValid > 0
        summary.percentExcitatory = 100 * summary.nExcitatory / nValid;
        summary.percentInhibitory = 100 * summary.nInhibitory / nValid;
    else
        summary.percentExcitatory = 0;
        summary.percentInhibitory = 0;
    end
if summary.nInhibitory > 0
    summary.EIRatio = summary.nExcitatory / summary.nInhibitory;
else
    summary.EIRatio = Inf; % Assign Infinity if no inhibitory neurons are found
end
    
    classification.summary = summary;
        
    fprintf('Classification Results:\n');
    fprintf('  Excitatory: %d (%.1f%%)\n', classification.summary.nExcitatory, classification.summary.percentExcitatory);
    fprintf('  Inhibitory: %d (%.1f%%)\n', classification.summary.nInhibitory, classification.summary.percentInhibitory);
end
    


%[text] **--- Other Helper Functions for the E/I system ---**

function modalityTest = testBimodality(data, ~)
    % Simplified bimodality test using Hartigan's Dip Test
    modalityTest = struct();
    %   The function removes any NaN (not-a-number) or infinite values
    %   from the dataset to ensure that only valid data points are analyzed.
    data = data(~isnan(data) & isfinite(data));
    %   If the cleaned dataset contains fewer than 10 data points, the function 
    %   sets the modality type to 'Insufficient_Data' and assigns a p-value of 1, indicating that no meaningful test can be performed.
    if length(data) < 10
        modalityTest.modalityType = 'Insufficient_Data';
        modalityTest.pValue = 1;
        return;
    end
    
    % NOTE: hartigansdipsigniftest is from a File Exchange toolbox.
    % If you don't have it, this will error. A basic check can be substituted.
    % as of 09/17/25 this has been downloaded.
    try
        [~, pValue] = hartigansdipsigniftest(data, 500); 
        modalityTest.pValue = pValue;
        if pValue < 0.05
            modalityTest.modalityType = 'Bimodal';
        else
            modalityTest.modalityType = 'Unimodal';
        end
    catch
        warning('Hartigan''s Dip Test not found. Assuming unimodal distribution.');
        modalityTest.modalityType = 'Unimodal';
        modalityTest.pValue = 1;
    end
end
%[text] Commenting resumes here:
function combinedAnalysis = analyzeCombinedFeatures(~, ~)
    % Placeholder, can be expanded for 2D analysis
    combinedAnalysis.prefersBimodal2D = false;
end

%   If the analysis is valid and the spike width is classified as bimodal, it selects a strategy specific to bimodal distributions;
%   otherwise, it defaults to a percentile-based strategy. 
function strategy = determineDistributionStrategy(distAnalysis)
    % Simplified strategy determination
    if ~distAnalysis.isValid
        strategy = 'insufficient_data';
        return;
    end
    
    if strcmp(distAnalysis.spikeWidth.modalityType, 'Bimodal')
        strategy = 'spike_width_bimodal';
    else
        strategy = 'percentile_based';
    end
end


%    If no valleys are found, it defaults to the median of the data. 
function threshold = findOptimalBimodalThreshold(data, ~)
    % Find valley between two modes from kernel density estimate
    [density, xi] = ksdensity(data);
    [~, locs] = findpeaks(-density); % Find valleys
    
    if ~isempty(locs)
        threshold = xi(locs(1)); % Use the first significant valley
    else
        threshold = median(data); % Fallback %can change this to whatever
    end
end


%   The calculateAsymmetryIndex function takes a waveform and the index of a trough 
%   as inputs, and computes the asymmetry of the waveform around that trough.
function asymmetry = calculateAsymmetryIndex(waveform, troughIdx)
    if troughIdx <= 1 || troughIdx >= length(waveform)
        asymmetry = 0;
        return;
    end
    %   The areas before and after the trough are calculated. preTroughArea sums the absolute differences 
    %   between the waveform values before the trough and the trough value itself. 
    %   Similarly, postTroughArea does the same for the values after the trough.
    preTroughArea = sum(abs(waveform(1:troughIdx-1) - waveform(troughIdx)));
    postTroughArea = sum(abs(waveform(troughIdx+1:end) - waveform(troughIdx)));
    totalArea = preTroughArea + postTroughArea;
    if totalArea > eps
        asymmetry = (postTroughArea - preTroughArea) / totalArea;
    else
        asymmetry = 0;
    end
end

%   assesses the quality of a classification by computing the silhouette score, which indicates 
%   how similar an object is to its own cluster compared to other clusters.
%   If most points in a dataset have high silhouette values (typically close to 1), it suggests that the clustering solution is appropriate. 
%   Conversely, if many points have low or negative silhouette values, it may indicate that the clustering solution is suboptimal, possibly due to too many or too few clusters.
function validationResults = validateClassificationAccuracy(classification, features)
    % Validates classification quality using silhouette score.
    
    fprintf('Calculating silhouette score to validate cluster separation...\n');
    validationResults = struct('meanSilhouetteScore', NaN, 'individualScores', []);

    % Get the indices of all classified neurons (E or I)
    validIdx = classification.isExcitatory | classification.isInhibitory;
    
    if sum(validIdx) < 10
        fprintf('Warning: Not enough classified neurons to calculate a meaningful silhouette score.\n');
        return;
    end
    
    % Get the feature data and group labels for classified neurons
    % Using Spike Width as the primary feature for clustering
    featureData = features.spikeWidth(validIdx);
    
    % Create a group vector: 1 for Inhibitory, 2 for Excitatory
    groups = ones(sum(validIdx), 1);
    groups(classification.isExcitatory(validIdx)) = 2;

    % Calculate the silhouette score
    try
        scores = silhouette(featureData, groups);
        validationResults.meanSilhouetteScore = mean(scores);
        validationResults.individualScores = scores;
        fprintf('  Mean Silhouette Score: %.3f\n', validationResults.meanSilhouetteScore);
    catch ME
        fprintf('Warning: Could not compute silhouette score. %s\n', ME.message);
    end
end


function summary = generateClassificationSummary(classification, validationResults)
    % Generates a comprehensive summary of classification and validation.
    
    fprintf('Generating comprehensive summary...\n');
    
    % Start with the summary from the classification step
    summary = classification.summary;
    
    % Add the validation results to the summary
    if isfield(validationResults, 'meanSilhouetteScore')
        summary.meanSilhouetteScore = validationResults.meanSilhouetteScore;
    else
        summary.meanSilhouetteScore = NaN;
    end
end

%function printSummaryResults(~)
 %   fprintf('Placeholder: Skipping final summary printout...\n');
%end
function printSummaryResults(summary)
    % Prints the final, formatted summary of the classification results.
    
    if isempty(summary) || ~isstruct(summary)
        fprintf('\n--- No summary data to print. ---\n');
        return;
    end
    
    % Calculate total classified neurons for context
    nValid = summary.nExcitatory + summary.nInhibitory;

    fprintf('\n--- Final Classification Summary ---\n');
    fprintf('Total Neurons Analyzed: %d\n', nValid);
    fprintf('----------------------------------\n');
    fprintf('Excitatory (Putative):   %d (%.1f%%)\n', summary.nExcitatory, summary.percentExcitatory);
    fprintf('Inhibitory (Putative):   %d (%.1f%%)\n', summary.nInhibitory, summary.percentInhibitory);
    fprintf('E/I Ratio:               %.2f\n', summary.EIRatio);
    fprintf('----------------------------------\n');
end

% --- Plotting Function for E/I System ---

%   SCATTER PLOT AND HISTOGRAM
%    produces two types of plots: a scatter plot showing the relationship between spike width and peak-to-trough ratio, 
%    and a histogram displaying the distribution of spike widths for both excitatory and inhibitory neuron clusters.
function figures = generateEnhancedClassificationPlots(features, classification)
    figures = struct();
    validIdx = features.validNeurons;
    
    if sum(validIdx) == 0, return; end

    % --- Create the Main Scatter Plot ---
    figures.mainScatter = figure('Name', 'E/I K-Means Clustering Results');
    hold on;
    
    spikeWidths = features.spikeWidth;
    ratios = features.peakToTroughRatio;
    isExc = classification.isExcitatory;
    isInh = classification.isInhibitory;
    
    % Plot excitatory and inhibitory clusters
    %   The function plots the excitatory neurons in blue and inhibitory neurons in red using scatter plots. 
    %   Each point represents a neuron, with its position determined by its spike width and peak-to-trough ratio.
    scatter(spikeWidths(isExc), ratios(isExc), 30, 'b', 'filled', 'DisplayName', 'Excitatory (putative)');
    scatter(spikeWidths(isInh), ratios(isInh), 30, 'r', 'filled', 'DisplayName', 'Inhibitory (putative)');
    
    % Plot the centroids (cluster centers)
%In the context of a neural signal processing pipeline, centroids play a significant role in clustering analysis, particularly when using algorithms like k-means clustering. Here's how centroids are relevant:
%Definition of Clusters: Centroids represent the average position of all data points (neurons, in this case) within a cluster. They serve as the central reference points that define the characteristics of each cluster, such as excitatory and inhibitory neurons.
%Data Segmentation: By calculating centroids, you can effectively segment the neural signals into distinct groups based on their features (e.g., spike width, peak-to-trough ratio). This segmentation is crucial for understanding the different types of neural activity and their implications in neuroscience.
%Visualization and Interpretation: Centroids facilitate the visualization of clustering results. In plots, centroids can be marked to show the center of each cluster, helping researchers interpret the clustering outcomes and assess the separation between different neuron types.
%Iterative Improvement: During the clustering process, centroids are recalculated iteratively as data points are assigned to clusters. This iterative refinement helps improve the accuracy of the clustering, ensuring that the final clusters are representative of the underlying data structure.
%Feature Extraction: In a broader signal processing context, centroids can also be used in feature extraction, where they help summarize the characteristics of the data, making it easier to analyze and classify neural signals.

    if isfield(classification, 'centroids')
        centroids = classification.centroids;
        plot(centroids(:,1), centroids(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3, 'DisplayName', 'Cluster Centroids');
    end

    xlabel('Spike Width (ms)');
    ylabel('Peak-to-Trough Ratio');
    title(['E/I Classification using K-Means (k=2)']);
    legend('show', 'Location', 'best');
    grid on;
    hold off;

    % --- Spike Width Histogram ---
%   A new figure for the histogram is created. It plots the distribution of spike widths for both neuron types, allowing for a comparison of their characteristics.

    figures.spikeWidthHistogram = figure('Name', 'Spike Width Distribution by K-Means Cluster');
    hold on;
    histogram(spikeWidths(isInh), 30, 'FaceColor', 'r', 'DisplayName', 'Inhibitory Cluster');
    histogram(spikeWidths(isExc), 30, 'FaceColor', 'b', 'DisplayName', 'Excitatory Cluster');
    title('Spike Width Distribution by Class');
    xlabel('Spike Width (ms)');
    ylabel('Count');
    legend('show');
    grid on;
    hold off;
    
    fprintf('Generated %d visualization figures.\n', numel(fieldnames(figures)));
end

function results = validateEIClassification(spikeData, classification, brainLocTable, peakChannels, areaName, numPermutations)
    % Validates E/I classification for a specific brain area using permutation testing.
    
    results = struct();
    
    % Step 1: Find neurons belonging to the specified brain area
    allBrainAreas = brainLocTable.allen_ontology(peakChannels + 1); % Get brain area for each cluster
    areaClusterIndices = find(strcmp(allBrainAreas, areaName));
    
    % Find the intersection with clusters that were actually analyzed in spikeData
    [~, analyzedIdx, ~] = intersect(spikeData.clusters, areaClusterIndices);
    
    if isempty(analyzedIdx)
        fprintf('No valid neurons found for brain area: %s. Skipping validation.\n', areaName);
        return;
    end
    
    % Get the E/I labels for these specific neurons
    isExcitatory = classification.isExcitatory(analyzedIdx);
    isInhibitory = classification.isInhibitory(analyzedIdx);
    
    % Step 2: Calculate the OBSERVED E/I ratio
    %   The function calculates the number of excitatory and inhibitory neurons and computes the observed E/I ratio, ensuring no division by zero.
    observed_E = sum(isExcitatory);
    observed_I = sum(isInhibitory);
    observedRatio = observed_E / max(1, observed_I);
    
    fprintf('Validating for area: %s\n', areaName);
    fprintf('  Observed E/I Ratio: %.2f (E=%d, I=%d)\n', observedRatio, observed_E, observed_I);
    
    results.observedRatio = observedRatio;
    results.observedE = observed_E;
    results.observedI = observed_I;
    
    % Step 3: Create a null distribution via permutation testing
    fprintf('  Running %d permutations...\n', numPermutations);
    
    allLabels = [repmat('E', observed_E, 1); repmat('I', observed_I, 1)];
    nullRatios = zeros(numPermutations, 1);
    %   The loop shuffles the labels and recalculates the E/I ratio for each permutation, storing the results.
    for i = 1:numPermutations
        % Shuffle the 'E' and 'I' labels randomly
        shuffledLabels = allLabels(randperm(length(allLabels)));
        
        % Recalculate the ratio with the shuffled labels
        shuffled_E = sum(shuffledLabels == 'E');
        shuffled_I = sum(shuffledLabels == 'I');
        nullRatios(i) = shuffled_E / max(1, shuffled_I);
    end
    
    results.nullDistribution = nullRatios;
    
    % Step 4 & 5: Compare observed to null and calculate p-value
    % p-value is the proportion of null ratios that are more extreme than the observed ratio
    pValue = sum(nullRatios >= observedRatio) / numPermutations;
    
    fprintf('  Permutation Test P-value: %.4f\n', pValue);
    results.pValue = pValue;
    
    if pValue < 0.05
        fprintf('  Result is statistically significant. The observed E/I ratio is unlikely to be due to chance.\n');
    else
        fprintf('  Result is not statistically significant. The observed E/I ratio could be due to chance.\n');
    end
    
    % Step 6: Generate visualization
    results.figure = figure('Name', ['Permutation Test for E/I Ratio in ' areaName]);
    histogram(nullRatios, 50, 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7], 'DisplayName', 'Null Distribution (Shuffled)');
    hold on;
    xline(observedRatio, 'r-', 'LineWidth', 3, 'DisplayName', sprintf('Observed Ratio = %.2f', observedRatio));
    title(['Permutation Test for E/I Ratio in ' areaName]);
    xlabel('E/I Ratio');
    ylabel('Probability Density');
    legend('show');
    grid on;
end
%[text] 
%%
%[text] ## --- Step 10: Spike Timing Correlation Analysis System ---

function correlationResults = runSpikeCorrelationAnalysis(v1Data, enhancedResults, clusterPeakChan, rawRow)
    % Main function to orchestrate the entire spike correlation analysis.
    
    % --- Analysis Parameters ---
    p = struct();
    p.binSize = 0.001;       % 1 ms bin size for correlograms
    p.windowSize = 0.100;    % 100 ms window on each side (total 200 ms)
    p.asymmetryWindow = 0.010; % 10 ms for asymmetry index AUC
    p.jitterWindow = 0.020;  % +/- 10 ms jitter window for correction
    p.nJitters = 100;        % Number of jitters for baseline
    
    fprintf('Assigning cortical layers based on channel depth...\n');
    layers = assignCorticalLayers(v1Data.clusters, clusterPeakChan, rawRow);
    
    % --- Initialization ---
    nClusters = v1Data.nClusters;
    clusterIDs = v1Data.clusters;
    isExc = enhancedResults.enhancedClassification.isExcitatory;
    isEnh = enhancedResults.enhancedClassification.isInhibitory;
    
    correlationResults = struct();
    correlationResults.params = p;
    correlationResults.layers = layers;
    correlationResults.acg = cell(nClusters, 1);
    correlationResults.ccg = cell(nClusters, nClusters); % This will store corrected CCGs
    correlationResults.lags = [];
    correlationResults.asymmetryIndex = nan(nClusters, nClusters);
    
    fprintf('Calculating auto- and cross-correlograms for %d neurons...\n', nClusters);
    % In function runSpikeCorrelationAnalysis, before the nested CCG loops:

% Add Memory Monitoring
%meminfo = memory;
%if meminfo.MemAvailableAllArrays < 1e9 % Less than 1GB available
%    warning('Low memory detected. Consider reducing analysis scope for region %s.', v1Data.name);
    % You might want to skip the CCG calculation if memory is too low
    % return;
%end
    % --- Main Calculation Loop ---
MAX_CCG_PAIRS = 10000; % Limit total CCG calculations
total_pairs = nClusters * (nClusters - 1) / 2;

if total_pairs > MAX_CCG_PAIRS
    fprintf('Too many potential pairs (%d). Skipping CCG calculation for region %s.\n', ...
            total_pairs, v1Data.name);
    % You could also implement random sampling of pairs here instead of skipping
    return; % Exit the function if too many pairs
end
    
    for i = 1:nClusters
        spikeTimes1 = v1Data.spikeTimes(v1Data.spikeClusterIDs == (clusterIDs(i) - 1));
        
        % Calculate Auto-Correlogram (ACG)
        [acg, lags] = calculateCorrelogram(spikeTimes1, spikeTimes1, p.binSize, p.windowSize);
        correlationResults.acg{i} = acg;
        % STORE THE LAGS VECTOR (add this)
        if ~isfield(correlationResults, 'lags')
            correlationResults.lags = lags;
        end
        
        % Calculate Cross-Correlograms (CCG) for pairs
        for j = (i+1):nClusters
            spikeTimes2 = v1Data.spikeTimes(v1Data.spikeClusterIDs == (clusterIDs(j) - 1));
            
            % 1. Calculate the RAW CCG
            [raw_ccg, lags_for_asymmetry] = calculateCorrelogram(spikeTimes1, spikeTimes2, p.binSize, p.windowSize);
            
            % 2. --- JITTER CORRECTION LOGIC ---
            jittered_ccgs = zeros(p.nJitters, length(raw_ccg));
            for k = 1:p.nJitters
                jitter = (rand(size(spikeTimes1)) - 0.5) * p.jitterWindow;
                jittered_spikeTimes1 = spikeTimes1 + jitter;
                [jittered_ccgs(k,:), ~] = calculateCorrelogram(sort(jittered_spikeTimes1), spikeTimes2, p.binSize, p.windowSize);
            end
            
            % 3. Calculate baseline and subtract it
            mean_jitter_ccg = mean(jittered_ccgs, 1);
            corrected_ccg = raw_ccg - mean_jitter_ccg;
            
            % 4. Store the CORRECTED CCG for plotting
            correlationResults.ccg{i, j} = corrected_ccg; 
            correlationResults.ccg{j, i} = corrected_ccg; 
            
            % 5. Asymmetry analysis on the RAW CCG
            if strcmp(layers{i}, 'L4') && strcmp(layers{j}, 'L2/3')
                idx = calculateCCGAsymmetry(raw_ccg, lags_for_asymmetry, p.asymmetryWindow);
                correlationResults.asymmetryIndex(i, j) = idx;
            end
        end
    end
    
    fprintf('Correlation analysis complete. Generating visualizations...\n');
    
    % --- Visualization ---
    correlationResults.figures = plotCorrelationResults(correlationResults, isExc, isEnh);
    
end

    function [counts, lags] = calculateCorrelogram(spikeTimes1, spikeTimes2, binSize, windowSize)
    % Efficiently calculates a raw auto- or cross-correlogram.
    % This is a general-purpose tool.
    
    nSpikes1 = length(spikeTimes1);
    nSpikes2 = length(spikeTimes2);
    
    if nSpikes1 == 0 || nSpikes2 == 0
        edges = -windowSize - binSize/2 : binSize : windowSize + binSize/2;
        lags = edges(1:end-1) + binSize/2;
        counts = zeros(size(lags));
        return;
    end
    
    % Vectorized implementation is much faster than nested loops
    [I, J] = find(abs(spikeTimes1(:) - spikeTimes2(:)') <= windowSize);
    
    if isempty(I)
       edges = -windowSize - binSize/2 : binSize : windowSize + binSize/2;
      lags = edges(1:end-1) + binSize/2;
      counts = zeros(size(lags));
      return;
    end
    
    % Calculate time differences for valid pairs
    timeDiffs = spikeTimes2(J) - spikeTimes1(I);
    
    % If it's an auto-correlogram, remove the zero-lag bin (self-correlation)
    isACG = isequal(spikeTimes1, spikeTimes2);
    if isACG
        timeDiffs(timeDiffs == 0) = [];
    end
    
    % Define histogram edges and compute the correlogram
    edges = -windowSize - binSize/2 : binSize : windowSize + binSize/2;
    counts = histcounts(timeDiffs, edges);
    lags = edges(1:end-1) + binSize/2;
    
    % Normalize to firing rate (spikes/sec) based on the number of reference spikes
    if nSpikes1 > 0
        counts = counts ./ (binSize * nSpikes1);
    else
        counts = zeros(size(lags));
    end
end

function layers = assignCorticalLayers(clusterIDs, clusterPeakChan, rawRow)
    % Assigns a putative cortical layer based on channel depth (rawRow).
    % NOTE: These boundaries are estimates for a typical 384-channel probe
    % MAY NEED TO CHANGE^^^
    % and may need adjustment based on probe geometry and insertion angle.
    %   takes in cluster identifiers, their corresponding peak channel indices,
    %   and a depth mapping (rawRow) to categorize each cluster into a cortical layer
    
    nClusters = length(clusterIDs);
    layers = cell(nClusters, 1);
    
    % Get the peak channel for the clusters we are analyzing
    peakChannels = clusterPeakChan(clusterIDs);
    
    % Get the row (proxy for depth) for each peak channel
    % Channel indices are 1-based in MATLAB, rawRow is 0-based
    depths = rawRow(peakChannels + 1);
    %   This loop iterates through each cluster, checking the depth d and assigning the appropriate cortical layer based on predefined depth ranges. 
    %   The layers are categorized as L2/3, L4, L5, or L6, depending on the depth.
    for i = 1:nClusters
        d = depths(i);
        if d >= 0 && d < 100
            layers{i} = 'L2/3'; % Superficial layers
        elseif d >= 100 && d < 180
            layers{i} = 'L4';     % Input layer
        elseif d >= 180 && d < 280
            layers{i} = 'L5';     % Output layer
        else
            layers{i} = 'L6';     % Deepest layer
        end
    end
end

function asymmIndex = calculateCCGAsymmetry(ccg, lags, asymmetryWindow)
    % Calculates the asymmetry index of a cross-correlogram to infer connection directionality.
    % Formula: (Post - Pre) / (Post + Pre)
    
    % Find bins corresponding to the pre-synaptic window [-10ms, 0ms]
    pre_mask = lags >= -asymmetryWindow & lags < 0;
    
    % Find bins corresponding to the post-synaptic window [0ms, 10ms]
    post_mask = lags > 0 & lags <= asymmetryWindow;
    
    % Calculate the area under the curve (AUC) by summing bin heights
    auc_pre = sum(ccg(pre_mask));
    auc_post = sum(ccg(post_mask));
    
    total_auc = auc_pre + auc_post;
    
    if total_auc > 0
        asymmIndex = (auc_post - auc_pre) / total_auc;
    else
        asymmIndex = 0; % No activity in the window
    end
end

function figures = plotCorrelationResults(results, isExc, isEnh)
    % Generates visualizations for the correlation analysis.
    lags_ms = results.lags * 1000; % Convert lags to ms for plotting
    
    figures = struct(); 
    
    % --- 1. Plot Example Auto- and Cross-Correlograms ---
    figures.exampleCorrelograms = figure('Name', 'Example Spike Correlograms', 'Position', [100, 100, 1200, 500]);
    
    % Find an example bursting neuron (high ACG peak near 0)
    acg_peaks = cellfun(@(x) max([-inf, x(abs(lags_ms*1000) < 10 & abs(lags_ms*1000) > 2)]), results.acg);
    [~, bursty_idx] = max(acg_peaks);
    
    % Plot ACG of the bursting neuron
    subplot(1, 3, 1);
    bar(lags_ms, results.acg{bursty_idx}, 'k');
    xlim([-50, 50]);
    xlabel('Time Lag (ms)');
    ylabel('Spike Rate (Hz)');
    title(sprintf('ACG: Bursting Neuron (Cluster %d)', bursty_idx));
    grid on;
    
    % Find an example E->I pair (find a CCG with a sharp peak just after 0)
    peak_vals = zeros(size(results.ccg));
    for i=1:size(results.ccg,1)
        for j=1:size(results.ccg,2)
           if ~isempty(results.ccg{i,j}) && isExc(i) && isEnh(j)
               ccg = results.ccg{i,j};
               lags = results.lags;
               % Look for a peak in a short window after t=0
               peak_vals(i,j) = max(ccg(lags > 0.001 & lags < 0.005));
           end
        end
    end
    
    % --- ROBUSTNESS CHECK for E->I plot ---
    subplot(1, 3, 2);
    if any(peak_vals(:) > 0)
        [~, linear_idx] = max(peak_vals(:));
        [ei_idx1, ei_idx2] = ind2sub(size(peak_vals), linear_idx);
        
        bar(lags_ms, results.ccg{ei_idx1, ei_idx2}, 'k');
        xlim([-50, 50]);
        xlabel('Time Lag (ms)');
        ylabel('Spike Rate (Hz)');
        title(sprintf('CCG: Putative E -> I Connection\n(E Cluster %d -> I Cluster %d)', ei_idx1, ei_idx2));
        grid on;
    else
        text(0.5, 0.5, 'No clear E -> I pairs found.', 'HorizontalAlignment', 'center');
        title('CCG: Putative E -> I Connection');
        axis off;
    end
    
    % --- ROBUSTNESS CHECK for Asymmetry plot ---
    subplot(1, 3, 3);
    validAsymmIndices = results.asymmetryIndex(~isnan(results.asymmetryIndex));
    
    if ~isempty(validAsymmIndices)
        % Find the pair with the largest (most positive) asymmetry index
        [max_asymm_val, asymm_idx] = max(results.asymmetryIndex(:));
        [asymm_i, asymm_j] = ind2sub(size(results.asymmetryIndex), asymm_idx);
        
        bar(lags_ms, results.ccg{asymm_i, asymm_j}, 'k');
        hold on;
        win_ms = results.params.asymmetryWindow * 1000;
        fill([-win_ms 0 0 -win_ms], [0 0 max(ylim) max(ylim)], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Pre');
        fill([0 win_ms win_ms 0], [0 0 max(ylim) max(ylim)], 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Post');
        xlim([-30, 30]);
        xlabel('Time Lag (ms)');
        ylabel('Spike Rate (Hz)');
        title(sprintf('CCG: L4 -> L2/3 (Asymmetry = %.2f)', max_asymm_val));
        legend('show');
        grid on;
    else
        % If no L4->L2/3 pairs were found, display a message
        text(0.5, 0.5, 'No L4 -> L2/3 pairs found.', 'HorizontalAlignment', 'center');
        title('CCG: L4 -> L2/3');
        axis off;
    end
    
    % --- 2. Plot Asymmetry Index Matrix ---
    figures.asymmetryMatrix = figure('Name', 'L4 -> L2/3 Asymmetry Index');
    imagesc(results.asymmetryIndex, 'AlphaData', ~isnan(results.asymmetryIndex));
    colorbar;
    caxis([-1, 1]);
    colormap(redblue()); % Ensure you have the redblue helper function
    xlabel('Target Neuron (columns)');
    ylabel('Source Neuron (rows)');
    title('Asymmetry Index for L4 -> L2/3 Pairs (NaNs are transparent)');
    
    fprintf('Generated %d correlation visualization figures.\n', numel(fieldnames(figures)));
    
end

% A colormap that goes from red (negative) to white (zero) to blue (positive)
function cmap = redblue()
    m = 100;
    r = [(0:1:m-1)'/m; ones(m,1)];
    g = [(0:1:m-1)'/m; (m-1:-1:0)'/m];
    b = [ones(m,1); (m-1:-1:0)'/m];
    cmap = flipud([r g b]);
end

%% --- Step 11: Layer-Aware E/I Analysis Framework ---

function results = runLayerAwareEIAnalysis(v1Data, features, clusterPeakChan, rawRow)
    % Main function to analyze spike width distributions on a per-layer basis.
    % Processes data related to neurons in a specific brain region, assigning them to cortical layers and 
    % analyzing their spike width distributions. It performs several statistical tests, including Hartigan's Dip Test
    % and Gaussian Mixture Model (GMM) analysis, to determine the characteristics of the spike width distributions for
    % each layer. The results, including neuron counts and statistical measures, are stored in a table for easy interpretation 
    % and further analysis.

    fprintf('\n=== Layer-Aware E/I Analysis ===\n');
    
    % --- Assign Cortical Layers ---
    fprintf('Assigning cortical layers to all valid neurons...\n');
    layers = assignCorticalLayers(v1Data.clusters, clusterPeakChan, rawRow);
    
    uniqueLayers = unique(layers);
    nLayers = length(uniqueLayers);
    
    % --- Initialize Results Table ---
    % Using a table is great for summarizing mixed data types.
    resultsTable = table('Size', [nLayers, 6], ...
        'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'logical'}, ...
        'VariableNames', {'Layer', 'NeuronCount', 'Dip_pValue', 'GMM_SilhouetteScore', 'GMM_Threshold_ms', 'IsBimodal'});
    
    allLayerData = struct(); % To store detailed results for plotting
    
    fprintf('Analyzing spike width distribution for each layer:\n');
    
    % --- Loop Through Each Layer ---
    for i = 1:nLayers
        currentLayer = uniqueLayers{i};
        
        % --- Sanitize the layer name for use as a struct field ---
        validFieldName = matlab.lang.makeValidName(currentLayer);
        
        layerMask = strcmp(layers, currentLayer);
        
        % Get spike widths for neurons only in the current layer
        validMask = features.validNeurons & ~isnan(features.spikeWidth);
        layerSpikeWidths = features.spikeWidth(layerMask & validMask);
        
        neuronCount = length(layerSpikeWidths);
        fprintf('  - Layer %s: Found %d valid neurons.\n', currentLayer, neuronCount);
        
        % Initialize results for this layer
        dipPValue = NaN;
        silhouetteScore = NaN;
        gmmThreshold = NaN;
        isBimodal = false;
        
        % Only proceed if there are enough neurons for meaningful analysis
        if neuronCount > 10 % Minimum for GMM and dip test
            % --- Test 1: Hartigan's Dip Test ---
            dipTest = testBimodality(layerSpikeWidths, '');
            dipPValue = dipTest.pValue;
            
            % --- Test 2: Gaussian Mixture Model (GMM) & Silhouette Score ---
            [gmm, gmmClassification] = fitGMM(layerSpikeWidths);
            silhouetteScore = calculateSilhouetteScore(layerSpikeWidths, gmmClassification.labels);
            
            % --- Test 3: Calculate Layer-Specific Threshold ---
            % A low p-value suggests the GMM separation is meaningful
            isBimodal = dipPValue < 0.05 && silhouetteScore > 0.4; % Stricter criteria
            if isBimodal
                gmmThreshold = findGMMThreshold(gmm, layerSpikeWidths);
            end
            
            % Store detailed data for plotting (using the sanitized field name)
            allLayerData.(validFieldName).spikeWidths = layerSpikeWidths;
            allLayerData.(validFieldName).gmm = gmm;
            allLayerData.(validFieldName).threshold = gmmThreshold;
            allLayerData.(validFieldName).gmmClassification = gmmClassification;
            allLayerData.(validFieldName).dipTest = dipTest;
            
        else
            fprintf('    ...Skipping due to insufficient neuron count.\n');
        end
        
        % Store summary results in the table
        resultsTable(i, :) = {currentLayer, neuronCount, dipPValue, silhouetteScore, gmmThreshold, isBimodal};
    end
    
    % --- Display Summary Table ---
    fprintf('\n--- Layer-Aware Bimodality Summary ---\n');
    disp(resultsTable);
    
    % --- Generate Visualization ---
    fprintf('Generating layer-stratified plots...\n');
    try
        fig = plotLayerAwareResults(resultsTable, allLayerData);
    catch ME
        fprintf('Warning: Could not generate layer plots: %s\n', ME.message);
        fig = [];
    end
    
    % --- Statistical Summary ---
    bimodalLayers = resultsTable.Layer(resultsTable.IsBimodal);
    if ~isempty(bimodalLayers)
        fprintf('\nLayers showing bimodal spike width distributions:\n');
        for j = 1:length(bimodalLayers)
            fprintf('  - %s\n', bimodalLayers{j});
        end
    else
        fprintf('\nNo layers showed clear bimodal distributions.\n');
    end
    
    % --- Compile Final Results ---
    results = struct();
    results.summaryTable = resultsTable;
    results.detailedData = allLayerData;
    results.figure = fig;
    results.bimodalLayers = bimodalLayers;
    results.layers = layers;
    results.uniqueLayers = uniqueLayers;
    
    fprintf('Layer-aware E/I analysis complete!\n');
end

% Helper function to fit GMM (moved from truncated area)
function [gmm_model, classification] = fitGMM(data)
    % Fits a 2-component Gaussian Mixture Model to 1D spike width data.
    classification = struct('labels', [], 'inh_mu', NaN, 'exc_mu', NaN);
    data = data(:);
    gmm_model = [];
    
    try
        options = statset('MaxIter', 1000);
        gmm_model = fitgmdist(data, 2, 'Options', options, 'RegularizationValue', 1e-5);
        
        % Classify each data point
        classification.labels = cluster(gmm_model, data);
        
        % Identify Inhibitory (narrow) vs Excitatory (wide) clusters
        means = gmm_model.mu;
        [narrow_mean, inh_idx] = min(means);
        [wide_mean, ~] = max(means);
        
        classification.inh_mu = narrow_mean;
        classification.exc_mu = wide_mean;
        
        % Ensure label 1 is always Inhibitory for consistency
        if inh_idx == 2
            % Swap labels if the second cluster was the narrow one
            new_labels = classification.labels;
            new_labels(classification.labels == 1) = 2;
            new_labels(classification.labels == 2) = 1;
            classification.labels = new_labels;
        end
        
    catch ME
        fprintf('    Warning: GMM fitting failed. %s\n', ME.message);
        % Create dummy classification for consistency
        classification.labels = ones(length(data), 1);
    end
end

% Helper function to calculate silhouette score
function score = calculateSilhouetteScore(data, labels)
    % Calculates the mean silhouette score for clustered data.
    score = NaN;
    if length(unique(labels)) > 1 && length(labels) > 2
        try
            score = mean(silhouette(data(:), labels));
        catch
            fprintf('    Warning: Could not compute silhouette score.\n');
        end
    end
end

% Helper function to find GMM threshold
function threshold = findGMMThreshold(gmm_model, data)
    % Finds the optimal threshold between two 1D Gaussian components.
    % This is the point where the posterior probabilities are equal.
    
    threshold = NaN;
    
    if isempty(gmm_model)
        return;
    end
    
    try
        % Create a high-resolution space between the two means
        means = sort(gmm_model.mu);
        x = linspace(means(1), means(2), 2000)';
        
        % Calculate the posterior probability for each component across this space
        posterior_probs = posterior(gmm_model, x);
        
        % The threshold is where the difference between probabilities is minimal
        [~, threshold_idx] = min(abs(posterior_probs(:,1) - posterior_probs(:,2)));
        threshold = x(threshold_idx);
    catch ME
        fprintf('    Warning: Could not find GMM threshold: %s\n', ME.message);
    end
end

% Helper function to plot layer-aware results
function fig = plotLayerAwareResults(summaryTable, allLayerData)
    % Visualizes spike width distributions and GMM fits for each layer.
    
    fig = figure('Name', 'Layer-Stratified Spike Width Analysis', 'Position', [50, 50, 1200, 800]);
    
    layersToPlot = summaryTable.Layer(summaryTable.NeuronCount > 10);
    nLayers = length(layersToPlot);
    
    if nLayers == 0
        text(0.5, 0.5, 'No layers with sufficient data for plotting', ...
             'HorizontalAlignment', 'center', 'FontSize', 14);
        axis off;
        return;
    end
    
    % Calculate subplot layout
    nCols = min(3, nLayers); % Maximum 3 columns
    nRows = ceil(nLayers / nCols);
    
    for i = 1:nLayers
        layerName = layersToPlot{i};
        subplot(nRows, nCols, i);
        hold on;
        
        % Sanitize the layer name for use as a struct field
        validFieldName = matlab.lang.makeValidName(layerName);
        
        % Check if this layer exists in our data
        if ~isfield(allLayerData, validFieldName)
            text(0.5, 0.5, sprintf('No data for %s', layerName), ...
                 'HorizontalAlignment', 'center');
            title(sprintf('Layer %s (No Data)', layerName));
            axis off;
            continue;
        end
        
        % Get data for this layer using the valid field name
        data = allLayerData.(validFieldName).spikeWidths;
        gmm = allLayerData.(validFieldName).gmm;
        threshold = allLayerData.(validFieldName).threshold;
        stats = summaryTable(strcmp(summaryTable.Layer, layerName), :);
        
        if isempty(data)
            text(0.5, 0.5, 'No spike width data', 'HorizontalAlignment', 'center');
            title(sprintf('Layer %s (No Data)', layerName));
            axis off;
            continue;
        end
        
        % Plot histogram
        try
            histogram(data, min(20, length(data)/3), 'Normalization', 'pdf', ...
                     'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k', 'DisplayName', 'Data');
        catch
            % Fallback to regular histogram if normalization fails
            histogram(data, min(20, length(data)/3), 'FaceColor', [0.7 0.7 0.7], ...
                     'EdgeColor', 'k', 'DisplayName', 'Data');
        end
        
        % Overlay GMM components if available
        if ~isempty(gmm)
            try
                x_range = linspace(min(data), max(data), 200)';
                % Get individual component PDFs, scaled by their proportion
                pdf1 = pdf('Normal', x_range, gmm.mu(1), sqrt(gmm.Sigma(1))) * gmm.ComponentProportion(1);
                pdf2 = pdf('Normal', x_range, gmm.mu(2), sqrt(gmm.Sigma(2))) * gmm.ComponentProportion(2);
                
                plot(x_range, pdf1, 'r-', 'LineWidth', 2, 'DisplayName', 'Inh (GMM)');
                plot(x_range, pdf2, 'b-', 'LineWidth', 2, 'DisplayName', 'Exc (GMM)');
            catch
                % Skip GMM overlay if it fails
            end
        end
        
        % Plot threshold line if available
        if ~isnan(threshold)
            y_lim = ylim;
            plot([threshold, threshold], y_lim, 'k--', 'LineWidth', 2, ...
                 'DisplayName', sprintf('Threshold = %.2fms', threshold));
        end
        
        xlabel('Spike Width (ms)');
        ylabel('Probability Density');
        title(sprintf('Layer %s (N=%d)\nDip p=%.3f, Sil=%.3f', ...
                     layerName, stats.NeuronCount, stats.Dip_pValue, stats.GMM_SilhouetteScore));
        
        % Add legend only if we have multiple elements
        if ~isempty(gmm) || ~isnan(threshold)
            legend('show', 'Location', 'best', 'FontSize', 8);
        end
        
        grid on;
        hold off;
    end
    
    % Add overall title
    sgtitle('Layer-Stratified Spike Width Distributions and E/I Classification', 'FontSize', 14);
end

% helper function to validate your input data 
function validateInputData(regionData)
    fprintf('Validating data for region: %s\n', regionData.name);
    assert(isfield(regionData, 'clusters'), 'Missing clusters field');
    assert(isfield(regionData, 'spikeTimes'), 'Missing spikeTimes field');
    assert(length(regionData.clusters) == regionData.nClusters, 'Cluster count mismatch');

    % Check for reasonable data ranges
    if any(regionData.spikeTimes < 0)
        warning('Found negative spike times in region %s', regionData.name);
    end
    fprintf('Validation successful.\n');
end
%[text] tabulate(enhancedResults.correlationAnalysis.layers)
%[text]   Value    Count   Percent
%[text]      L6       95    100.00%
%[text] 

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":31.2}
%---
%[output:4bf00665]
%   data: {"dataType":"text","outputData":{"text":"Step 1: Loading channel and probe metadata...\n","truncated":false}}
%---
%[output:2dccf135]
%   data: {"dataType":"text","outputData":{"text":"Found 11 unique brain regions to analyze.\n","truncated":false}}
%---
%[output:670b92fa]
%   data: {"dataType":"text","outputData":{"text":"    {'ACA' }\n    {'DP'  }\n    {'ILA' }\n    {'MOs' }\n    {'MRN' }\n    {'PL'  }\n    {'RSP' }\n    {'SCig'}\n    {'SCsg'}\n    {'TT'  }\n    {'VISp'}\n\n","truncated":false}}
%---
%[output:09e5d78f]
%   data: {"dataType":"text","outputData":{"text":"Step 6: Loading spike data...\n","truncated":false}}
%---
%[output:82297d39]
%   data: {"dataType":"text","outputData":{"text":"\nProcessing region 1\/11: ACA\n","truncated":false}}
%---
%[output:36ad48d2]
%   data: {"dataType":"text","outputData":{"text":"\n\n=======================================================\n","truncated":false}}
%---
%[output:2d636619]
%   data: {"dataType":"text","outputData":{"text":"Processing Brain Region: ACA (1 of 11)\n","truncated":false}}
%---
%[output:21c8206a]
%   data: {"dataType":"text","outputData":{"text":"=======================================================\n","truncated":false}}
%---
%[output:03cec9da]
%   data: {"dataType":"text","outputData":{"text":"Step 7: Filtering for ACA spikes...\n","truncated":false}}
%---
%[output:4bae02d6]
%   data: {"dataType":"text","outputData":{"text":"Created ACA data struct with 26 clusters and 233623 spikes.\n","truncated":false}}
%---
%[output:71a7f714]
%   data: {"dataType":"text","outputData":{"text":"Validating data for region: ACA\nValidation successful.\n","truncated":false}}
%---
%[output:19cb6f84]
%   data: {"dataType":"text","outputData":{"text":"\n\n=== Enhanced E\/I Classification Analysis ===\nExtracting comprehensive waveform features...\nExtracted features from 26\/26 valid neurons\nAnalyzing distribution characteristics...\n","truncated":false}}
%---
%[output:5a0f0243]
%   data: {"dataType":"warning","outputData":{"text":"Warning: Hartigan's Dip Test not found. Assuming unimodal distribution."}}
%---
%[output:7e23ba33]
%   data: {"dataType":"warning","outputData":{"text":"Warning: Hartigan's Dip Test not found. Assuming unimodal distribution."}}
%---
%[output:48b04b78]
%   data: {"dataType":"text","outputData":{"text":"Distribution Analysis Results:\n  Spike Width: Unimodal (p=1.0000)\n  Peak-Trough Ratio: Unimodal (p=1.0000)\n  Recommended Strategy: percentile_based\nDetermining optimal classification strategy...\nSelecting k-means clustering strategy with 2 features...\nApplying enhanced E\/I classification...\nRunning Gaussian Mixture Model (GMM) clustering on 26 neurons...\nClassification Results:\n  Excitatory: 9 (34.6%)\n  Inhibitory: 17 (65.4%)\nValidating classification accuracy...\nCalculating silhouette score to validate cluster separation...\n  Mean Silhouette Score: 0.109\nGenerating visualization plots...\n","truncated":false}}
%---
%[output:213832c2]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAhoAAAFECAYAAABh1Zc8AAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQncV9P2\/1eD8CoydqVIpgyRLjKmyZTUNZcyj7m6wqUMUamodJFCUS7+RAPJPNc1hQyFZPjxoyIuGTImbv\/fe1\/76Tzfvt\/v+Q7nPN99zl779Xpe9TznnH32Gvban7PW2mvXWrly5UrRphxQDigHlAPKAeWAciAGDtRSoBEDV7VL5YByQDmgHFAOKAcMBxRoqCIoB5QDygHlgHJAORAbBxRoxMZa7Vg5oBxQDigHlAPKAQUaqgPKAeWAckA5oBxQDsTGAQUasbFWO1YOKAeUA8oB5YByQIGG6oByQDmgHFAOKAeUA7FxQIFGbKzVjvNx4Pfff5dff\/1V6tatK2ussYYySzmgHFAOKAdSygEFGikVbE2T9e9\/\/1sGDBiQ87U77rij9O3bt+r6TTfdJFdffbWcdtpp8vHHH0u9evVk7NixOZ\/\/9NNPZfz48fL666\/LsmXLpFWrVrLXXntJjx49pHbt2ua5e+65R5555hnp2LGj+Xvc7bvvvjPjp914442y8cYbC3+79tprZf78+dKyZUtz\/dxzzzX3TJ06NZIhLV++XH755RfDs7XXXtv0ef7558uiRYvknHPOkbZt20bynrg6qamx\/utf\/5K77rpLNthgAxk+fHgVOQ8++KDwQ8ulKx988IHRT9vQ7c0339z8+vbbb8v1119fdW3EiBGy\/vrrx8WuSPq96KKL5MUXX5R7771XHn\/8cXn55ZflkEMOkc6dO5fc\/4IFC+SVV16RL774QrbZZhs5+OCDjT5OnjxZRo8eLaNGjZK999675P71wfRwQIFGemRZUUoACxjtXG3fffeVO+64w1ymRly7du1k8eLFMnTo0CqA8tFHH2V9\/LXXXpOTTjpJfvzxx9WuYyxZEDBwQ4YMkX\/+859y3HHHyRVXXBE7P7788kvZY489zHuee+45adKkiVnYLrvsMvO3Dh06GNo6depkfs9FX7EDtXSefPLJVe\/iHf\/7v\/8r1113nXTr1q3YLmv0\/poaK8DzkksukQ033FDmzJljaHziiSekd+\/e5v\/77befAHgtWAsyAZ07+uijq\/7EonnEEUeY3y1IthdZwDfZZJMa5WExLwMMALwZP3RccMEFct999xlw2qdPn2K6qrr30UcflbPPPrvas02bNjVABgDcunVr2WqrrYT78Fpq85sDCjT8ln9k1AeBBuChcePG1fpeb731jPGhWSN+2GGHGWOPwcu1EBNe4auLRZQFY9iwYbLuuuvK\/fffL1OmTDHP2UUgF9Cgj6+++sosBtb7kUn4Z599Zr5Ksy063JvtOv2+8cYbpqtddtlF1lxzTTnjjDPkqaeekhNOOEEuvfRS+c9\/\/iPvvPOOucfSb98NcKKPXF\/DX3\/9tTHS0Bts2YDGvHnzjJdjyy23NJ4V2\/B+fPLJJ+Ydwb9nEzzAKeyeYhTm559\/FmgAgAVbrrHae7755htZZ511si5Q33\/\/vaxYscJ4KcJaJtB4\/vnnjVxoeH3GjRuXU96ZQIOF+sorrzTPnnrqqTJz5syq12cCDfhIy8XLH374QX766Sdp1KhRXhLyySMXbzM7\/O233wzwfPfdd+X22283dEcBNHbffXdZunSp8C9gH28kv\/fs2dN8PABgHnnkERk0aFAVz8PkpdfTywEFGumVbY1SFgQaLLQseLkaC\/Ddd99tPBwAgHxAI\/gFyjPWg0Df1pgdcMABJqySCTRYsPAuYPBo9evXlzZt2hjjt9lmm5m\/YXxx9WKIaVwn1LHnnnuGXmfs3E\/DowEd9suZdxHa4Yu6a9eu5h4WL7724FX\/\/v2r7t1hhx2kS5cu5ku7Vq1a5ivw8ssvN4abxpfhMcccI6effroMHjzYjNk23s+CeuihhxpAg8ua97EQwQ+u2da8eXPj8dhpp53Mn1ggoPvMM8+Uhx9+2HiY+Col3GMX5EwZ2gUGzxFeKRr9AZqsfPDcsPAABmkAPGSEd4d8nMyxWrlB43vvvScAEfh3+OGHm2fgGYAFuQAWaNz74YcfGp7efPPNsv\/++6+mbkGgwT1HHnmkuScMZFhZWY8GABcA+uyzzwq5RYBKgBv8olmgwVj69etnQLGlG35bHSHU8Pe\/\/71K1+j3wAMPNDTSfyHyCONtJhMsuOJds2fPNuAtE2jccsstVaEgvDV4H\/M1wpg2PAdNAGzkfc0118h2221n5hs2ANCN7F944QWj19r85YACDX9lHynlQaCBwdxoo42q9c\/CBfhgQWJhwvC99NJLJlaeD2hYN3XQ\/Z1r4JlA4x\/\/+IfccMMN5nY8JywEvJ9FiYUHbwSLD4saX324mFm8+J1rb731Vt7rAJlg6IQ8jQceeMC8A3DAe1gQbegEo4xHhd\/tIoUh\/vzzz80YGS\/XWMho9IEniHHT6JtF7bbbbjPP2AUc8JEZjiAmbz0+AAy7+NEP8Xm+tuGJHQfvgnbbcoFFCyomTpxoQkM0CyonTZpkABphLhZlFtg\/\/elPVfkQ9us2c6yAqjvvvLNqccZjYUHWVVddJd27dzcg6+mnnzb3oAv2Or+HAQ3uQabIBVBHrkwuz5WlP+jRAAQCxOAbgAcPG2EIwg80ZEKz+QgWmFje8uz2229vgBMgius777xzlVdk4MCBcuKJJxYkjzDeZs4NeA6gP\/744w1IpQWBBnK3IRA8NoyRD4FcDb0B4ANqAS3WS2e9PIBcwC6eNGi2eku+kjZ\/OaBAw1\/ZR0p5WI4Gxo4vJRZLvkxJWuRfvnrzAQ27YO66666hyZSZQAMDTsiDL3QWvZEjRxrPB1\/tLIQY1gkTJpgFnTwPFmTc6TS+xgA5+a7jls7M0WDBwLthF1UW+CDQIEHR5gjMmjXLhBXgA4sRAA2DDWDBrU6YKOg1sd6KsBwNaLWLHrkq5KzghieHhsWWpFx+LNAghMXX6JIlS2SfffYx9OdavMOABouLXXwuvPBCOeqoowxo44dFnoUoF9BALiQq0vAmsJghOxZJ6z2xPGCRZ8EsFGgElf3JJ580MrcNGRHOsI3FlK9269FAj1ik4QkA5+KLLza6g7fKAo1p06YZHqJD0ACgBMQCLJAp+m71HD0kNET\/XLfhhjB54E0K423mpEa2b775pgEPjCMINACFgH3aeeedJ3\/7299MorUFutkMBPRZwGevW88RvzN\/8NLQLD3Qix5o85cDCjT8lX2klAeBBuAgM\/7MAoYBx7OBOxdjhdEKAxpBMMACka9lAg3i+Xz9AyoIEdhkUgs0+BJlEQ42DOJZZ51lxhZ2PVsyaBjQIOSA4WXRfeihh7KSw3tZuFiEgp6IQoEG+Qs29PHqq69W5TNkhprsQhBMILWhkTFjxphwTmYLAxosXnggbAiJ51nU4Skgg9BJLqARzIOwX93Qwf3wlWZd9eS24KanFeLRgN82V4bnSFq0Xg0rM0srIAxQbIEG9wIaAIjsrgLkAAwtf5AXnhe7kwXdoXEvOmfDW8jy\/\/2\/\/2c8Z3Ys3HfssccaUFmIPMJ4mykvK09AEKAjCDSC97Jba4sttjD5L9CWqzVo0KAqTEV4jtCk9ewQggNc2jwo+IdnqJyk00iNlHZWMQ4o0KgY69P14kJyNGxsN+idCAMauOPttlnr8recIy6MkecrGoOXCTT4UuSLzcbC2e3C15cFGvRDiADjjxci2Fjk+erMd53ky2I9GnhJ2BrJ9tzp06evpgSMl3HTWLTJ88AbxJd0oUCD\/AG7MONJaNiwoemPBZTF0IaO7MIWXKgLBRp4evCQ4NXZdtttTf82dAIAI7SC98qGhez7GUMuoIH3wnoJyHUAbAE0CNGwwyYINIKu+TCgQdgEPWEBtf3TL14nGu+yX\/b8zpc\/rn4LNAA36Bh6y1ZOGuDDyp6+SYAkN4Fwls3vscIltARAwGOFHAEifPWzBRrQnQk08skjjLeZCoVnCxnYEFQuoGF1guRlvG65GuAM+ggbnnLKKQYM07J5LWwuDh8eeAi1+csBBRr+yj5SyoNAY8aMGebrKLOxoLNDJGiUwoAGcW4WRJrdyspXMQDgr3\/9q\/k7X1F8MQeBBl\/v1uCzKOBytrkAFmhQ7wLwQ78s6Hxp2gWNRf1\/\/ud\/8l6n\/2KBBouddfnztQco4IsPPvBFT+IjwIJkO5I+Cf3Y5LxMoAGYwK1PCy7eLIg2WY+8D+LueHdYdPjCJiZPUmIpQIOkTnI5LM\/xXLCIWqDBFy9eJOL3LL6MH48J9Fm+Fws0oNOGn\/BwkfdC\/okFDWFAw+b3ADT56kZ3aEE3f6auBnM0SMC0X+fcx\/vhXxBokAiLF4h3ATzQUbwE6C\/AgpAEoQlAD3q21lprGcCHl6NQoNGsWbNQ3mbSgU7xPhZ6Fvwg0CDPhNoXFgQADtGdQkIn1juGTHmOOhrBBmDZeuutzZ+s3kZqcLSzRHFAgUaixOXuYMNyNDDy7PTA8M6dO7dqy2YY0IDiYFInhhojbZMBMXQs3ngfgkADUAF44D5cxlwH6ND46mQxwFPCVzh94ganX1vrg6868jnyXef+YoEGXhCAAws+Y+fHfk2TpEihKFv7gQWWRZEYO81u4wWosUiyqPXq1cvkeGQu3tZ7wXMsHsHQEV\/Rm266aUlAw4Ii+iUcwdZZG5KCV8jYghxbVI1cFN5vF9RigQaeh2AyaOYsKBRo8Bw5LyRzohfI\/bHHHltt+y33ZQINinKhDzS8A3hzgkADHgDCLL9J4rW5DITL4PdBBx1krgP0AGu82wIXCoqFAT8AQBhvM3lDSMYm7vJvEGjYkIYFI+hiMDk7m7Vh7rBjKhcYsWAyuDMFOq3Xy10LpiOLkwMKNOLkrkd9FwI02OJmiwZZ1tjkUH7PVdAKkAJIsK5u+yxGF3BhKzZao2oLdrFtla9quxDiErfGloQ9dsbwvN2Gafu1iZy4h\/NdD+Zo2MU7LEeDrYCAHL6sgwXIGCchE7586cO6pAElhAnwHNiMfq4RZrBueBY0u3jbr0f6AbDYrb3QBjDhq9t6euwzNgzCPWGhk4ULF5q8FrujAnkCyhiLDZ3wTrxFwV0s7LIAHLFQZY7V7owIfnUHQydchx4SW5EbAA8vDV4EeBjcAROccsifxM3MHUuM13quAGF4RzK3X2YCjeA2a7YfkwdjgQZAkZwkPBgswsEdMYBZQgw05GG3G7Mgk7tBfgPjQ3\/wrOHhyCePMN5mmhxLhwXXXLcJ1hZoUOnUFnmDX4C6fC1YjyTzPvsee0+25FGPzKKS+gcHEgk0MLxMblswxtZEyCZV9r3zBc2EwzDwlaUtmRwgH4Ay22wxROZhBY+gEl1hUcTtnKtCIYsYngTc\/uwC4d9gC7teCjehhUWbwk24nQEgwUbIAfd7rqJPPA8QItRiczCyjYOS6CxeLIwsbrkKlhVLA3F\/4vX53g1P2c2BnPiiL7WWAjK0W0jJnQAg8n67Q8ZWZS2Whjjut3JlzORmZMoVeaBP+WxWIeMqlLeEMPC0oAN2m20h\/Zd7DyE9Pg6CQKvcPvX55HIgcUCDSUpckZgzBizzCyAoCiYZX4kkXREv5EuQXQW4nrUpB5QDyeAAO0zwPOC94AuZ\/5MoSiukvkoyqIxvlLZ4lvWIxfem\/\/ZsPX14N\/C2hdUsiXs82n\/lOZA4oEHsnZhvMC6P69NWXQyylHMniH3aOKx1f2buXqi8GHQEygHlQD4OEDoi7GMrg3Iv4Ri2bearQqtc\/S8HSN4kZJQrJyVKPpEzg90NhoCi7F\/7Sh4HEgc0SJgiqYxdBjR7YBDxWOLLwUasm4QvqkPiAcElzt52Mv0zXZrJE52OWDmgHFAOKAeUA+5zIFFAw+6dD25Ls9UMqalAMlWwsZ2PbZaUkraFc0iiy3Yugvui0hEqB5QDygHlgHIgeRxIFNCwW6ZsOWvYTSW7Fi1arLZXm6p1O+64o5EIW9PYmYArj5hh2KFfttBTpnck10FTyRN79RGT95J5QmjSaco3fp\/o9YlWZO4TvT7R6ptsSZzOVosoqXY5UUDD1uEPFnwi857kMLZJBg\/usRUL7TkOCMgmKYXV3ifmm2urZVIFnW\/cJNTasslppC+TJp\/o9YlW5OwTvT7RqrJNtmVOFNCA1Zy1wCFL7LOn2ZLN7CzJzG4mdEL5YuorBIFG8GyHbOJToJFspQ4bvU8G2idadTEK0\/xkX\/dJl9NGa+KAhq3QR5lrtk+RTV2nTh1TLIhGwSJCJuRhkKVOBjTFg\/B2kMfBfezJ59lcTYFGsg1S2OjTNonVW7WKAyrbMO1P7nWVbXJllzigQe4FpyjaQ7Bw+bON1QIHPB6UZab6HQmgVOqz1RHZgUI5a3uMsQKN\/3LApwnsG70q2+Qa57CRq2zDOJTc62mTbeKAhlUddptwbkQhuQVsaSWXg+SaQra1qkcjuRO0kJGnbRKrR0M9GoXofdLv0XmbXAkmFmjEyXIFGnFyt\/J9q8GqvAziGoHKNi7OVr5flW3lZVDqCBRoZOGcAo1S1SkZz6nBSoacShmlyrYUriXjGZVtMuSUbZQKNBRoaI5Gcudv6Mh9Ms4wwyd6faJVZRs61Z2+QYGGAg2vjLMaLKftUdmD82nx9YlWnbdlT42KdqBAQ4GGAo2KTsF4X66LUbz8rWTvKttKcj\/ed6dNtgo0FGgo0IjXZlS097QZrDBm+kSvT7SqRyNM892+rkBDgYYCDbfnaFmj08WoLPY5\/bDK1mnxlDW4tMlWgYYCDQUaZZkEtx9Om8EK47ZP9PpEq3o0wjTf7esKNBRoKNBwe46WNTpdjMpin9MPq2ydFk9Zg0ubbBVoKNBQoFGWSXD74bQZrDBu+0SvT7SqRyNM892+rkBDgYb7QGPWLJH27SObST4ZaJ9o1cUosiniZEc+6XLaaFWgoUDDXaAxeLDIoEGrJMT\/Bw4s2wimbRLnY4hPtCrQKHtqON2BT7qcNloVaGSZWj033VQmffaZ05MuysE5qdSZIMMSHAHYcJLeKAUa6MsnWhVoxKREjnTrky6njVYFGsFJFNMXtCPzNOcwnFTqWrVys23lyrJY6iS9ZVGU+2GfaFWgEZMSOdKtT7qcNloVaNhJFOMXtCPzNDlAg5yMDh1ys23mzLJyNtI2iTV0sooDKlvXrU3p41PZls67Sj+pQMNKIMYv6EoLOez9Tk7gGOXhJL1hQirxuk+0qkejRCVJyGM+6XLaaFWgwSSL+Qva9XnspFLH6GFykt6YlMQnWhVoxKREjnTrky6njVYFGurR0F0njhjSOIaRNoMVxiOf6PWJVgWRYZrv9nUFGpqj4S7QsLLROholWxFdjEpmnfMPqmydF1HJA0ybbBVoBFVBd52UPDGS9GDaJnE+3vtEq371JmkWFj9Wn3Q5bbQq0Mii71pHo3gjkKQn0jaJFWis4oDKNkkzsbixqmyL45dLdyvQyCKNLbfcUj766COX5BTrWHyawPrVG6sqVbxzn3TZJ1p13lZ8apU1AGeBxscffywvv\/yyNGvWTFq3bi1rrrlmVkI\/\/\/xzWbBgQbVrm266qbRo0UJ+\/vlneemll6pdW3vttWXPPffMyzQFGmXplPMP+2SgfaJVFyPnp15ZA\/RJl9NGq5NA49Zbb5WhQ4fKdtttJ++++64BBhMnThRAQma7\/fbbZTC5FYF27LHHyrBhw+Ttt9+Wbt26VbvWtGlTefbZZxVoBDiQNqUOs2Y+0esTrQo0wjQ\/2dd90uW00eoc0Fi0aJG0a9dOhgwZIr169ZJPP\/1U2rZtKwMHDpQTTzxxtZnC32vXrm2uZ7ZHH31URo0aJU8\/\/XRRM0w9GkWxK3E3p20S5xOAT7Qq0EjcVCxqwD7pctpodQ5oPPjgg9K3b1+ZP39+lQejR48eUq9ePbnjjjtWU0zAR6dOnaRz586ycuVKadSoUdU948aNM16N4cOHy7fffiuNGzeWOnXqhCq3Ao1QFiX6hrRNYgUaqzigsk301Mw7eJVtcmXrHNAYPXq0PPTQQ\/Lkk09WcXXAgAEye\/bsrJ6JvffeW8jTsG2HHXaQ66+\/XgALF110kUyZMqXqWv369YX+O3bsGBo6ybwBQHPCCSckV9J5Rr548WIhpORL84len2hFf32i1ydafZPtsmXLpFWrVqkxyc4BDcDBe++9J9OnT69i8pgxY2Tq1Kmr5Vb89ttvBjR06dJFjjvuOHn\/\/ffl0ksvFQAFQAXPyA8\/\/GD+9uOPP5owynPPPWd+mjRpklOI6tFIjX5nJUS\/jNIrX59ku+See6Rxjx7pFWYGZT7JNm20Ogc0RowYIdOmTZM5c+ZUqRmJnQsXLpTx48eHTiqbSPrCCy+YUEmw2eRQvBpdu3ZVoPEHB9Km1GFK4hO9PtGK3L2gVwsLhk3xxF9Pmx47BzQmT54sF198scydO1fWXXddozDdu3cXQiR4KIKNWhfkdJx00knSsGFDc+muu+6Syy67TF588UWhL\/I3WrZsaa7hKSGXY8KECXnDJ+rRSPw8zUtA2iZxPmJ9otULoBHjYYOuz3qfdDlttDoHNL766itp06aNCYVccMEF8vzzz0ufPn3k7rvvlj322EPeeOMNeeKJJ+Scc84x82K\/\/fYzu1LwenANkPGnP\/1JJk2aJGeeeaapsXH\/\/ffL0qVLZeTIkaauBn1aYJJtcinQcN3klDe+tE1iBRqrOJB62daqlVvcK1eWNzEcfzr1sg3wP220Ogc04PUzzzwjp512WhXb+\/XrJ7179za\/AyBIDgVUABYeeeQRs6uExCjazjvvLGPHjjXJjXgw+vfvL2+++aa5tuGGG5pkULwj+ZoCDcctTpnDS9skVqDhCdDgcMEOHXKLe+ZMkfbty5wd7j6u89Zd2YSNzEmgwaB\/\/fVX+fDDD2WLLbbIWqgrkzDCKA0aNKi2vdXes2TJElm+fLkBH3Xr1g3jidmxoiXIQ9mU2BvUYCVWdKEDT71s1aMRqgNpuCFteuws0KiksijQqCT343932iaxejQ88WhApuZoxG8gHHhD2myUAo0sSqVAw4GZFuMQ0jaJFWh4BDSygY1Bg0SyVEaOcQpVpGudtxVheyQvVaChQMOPLYEBOavBisR2ONmJT7LVOhpOqmAkg0qbHivQUKChQCMS0+BmJ2kzWGFc9olen2hF7j7RmzZaFWgo0PBqAqvBCluqQ66z88HhnQ1pM9AaFvMsLPYHuWnTYwUaCjQUaJS59rr8eGQGKyHVKCOj12WhpnQxCmO5yjaMQ+5eV6ChQEOBhrvzs+yRRWKcE7TTIRJ6y+Z6zXTgE63qiawZnYrrLQo0FGgo0IhrdjnQbySLUYJqN0RCrwNyK2QIPtGqQKMQjXD3HgUaCjQUaLg7P8seWdmLUcKqUZZNb9kcr7kOfKJVgUbN6VUcb1KgoUBDgUYcM8uRPiNZjNSj4Yg0qw8jEtk6SVn2QflEb9poVaChQEOBRoKMbbFDjcRgpTBHo2fPnuaARW1+cmDPPfc052a52iKZtw4Rp0BDgYb7QCPiLZVpm8T57ElktKZs14lv1X8dWnOcGIrr8o9s3jrBbREFGgo03AUaMS1uaZvENQI07EsiBn1R28FCZev6QhM1X7S\/6hxwXf6F6nFS5KpAQ4GGm0AjRnd92iZxjQINxy1bobJ1faFxnM2JH57r8i9Uj5MiCAUaCjTcBBoxJiCmbRIr0FjFgUJl6\/pC49oCsnz5cvnll1+kYcOGrg0t63i++eYbqV+\/vtSrVy\/rddflX6geJ0IY\/3fosAINBRruAY2Yt1SmbRIr0IgeaOBQo7Vrl7\/i+s8\/\/yw77rhjVhGMHTtWDjnkkKLWgueff17OO+88mTNnjnz66acyb9680D4Kva+ogWTcfMEFF0iHDh2kS5cuJXXz2GOPyXbbbSdbbLFF3ueD91133XWyePFiGTVqVNHvnDZtmrz33nty6aWXKtAomnvRP6BAQ4GGe0ADmahHI5LZ7hOogmGF0pvrizYbxuVoF05hz3bEiwUa48aNk+23376azDbYYAPzVV1M++677+Sjjz6S1q1by1NPPSVXXHGFPPvss3m7KPS+YsYRvBfQc8kll8jjjz8utWvXLqkbAFefPn1CQVPwvkWLFsmvv\/4qW221VdHv\/O2336Rjx45yww03yE477bTa8+rRKJqlZT2gQEOBRsHGuSxNK\/ZhzdEolmNZ7y904Y3kZQ50Uii9uRaaDh1EABvZ2sqVq\/\/VAo3p06dLq1atVrvhyiuvlCVLlsjo0aNl5cqV0rdvX2nZsqX07t1bHn74Ybnmmmtk2bJlZgFmMX\/33XflxhtvlAEDBkivXr3MF\/3BBx9s\/vbEE0\/IxIkTDRBp27atXHjhhcKCmnkfwGPEiBHy+eefy1577SWDBg2STTfdVKZOnSoLFy6UL774Qn7\/\/Xf58MMPZejQoWY8NO7bfPPN5ZRTTqlGx5lnnil77723nHjiiYLH5aGHHpK1117b\/AsIgEb4eccddxhaABS022+\/XX766Sfzw4K\/ySabyMiRI2Xfffc19MCz\/\/znP9K+fXtDOx6M4H14av7973+be+j3sssuM\/2+\/\/770q9fP\/M+QjqDBw+Wl19+2QA9\/m7pGTNmjPzP\/\/yP4X1mU6BRs5NVgYYCDTeBBnLRXSdlW4NCF96yX+RIB4XSm22hCYvYDRr0X89GsFmgAYBo3rx51aVatWpJ165dDSjYf\/\/9zQLLojh8+HB5+umnhRyCzp07y8CBA004Ac\/F2WefLY0aNTKhE7wY\/\/znP6t+tt12W9lnn33kb3\/7mwE0\/\/jHP2Trrbc2YCN4HzkJAJOTTjrJvJ\/wDYv0lClTzP+vvfZaA1IAD1dffbW0a9dOzj33XPnhhx9k5513ljvvvNOAimDjdwsQHnzwQQOWjjnmGDnyyCPl5ptvFmjlX8ANdEEjjd9\/\/PFHMxZ+uP\/44483QOeEE06QYcOGyXrrrWfoHjJkiEBj8D6ACl6NAw88UP7617\/K\/PnzDcABQAB47rnnHjn88MNl3XVo7M4QAAAgAElEQVTXlTPOOENeeOEFGT9+vMydO9f87dFHHzXAg9\/r1KlTjSYFGjU7YRVoKNBwF2hY2US8pbLQxahmp2I8b\/OJVjhYKL3ZFppcTjQrmXxAo2nTpmZxs61u3bpy\/\/33m19vuukm8wVPA1CwOLIIv\/766zJ58mTzdxZOvuCbNGlSlaMRDImwgFNgDHDy1VdfGdDAwouXIngfgGD27NnGW0DDa3HAAQfIc889J\/fdd5\/xAvD1TwiEhRxg8eSTTxpvyd\/\/\/nd54403hLHbZgEI4AggZYHGggULZM011zRjBrgwluuvvz4r0IDmYEjkgw8+MOBnl112MV4XgAZeDQBP8D5AEUADQEIeDN4c8kSg57TTTpMddthB\/vKXvxhQBv\/xGLVp08bwGD4xRnJK4C0enWBToBGPvcnVqwINBRoFG+eaVc343lboYhTfCGquZ59ojRtozJy5ep5GWOiEMdl7CB3w1Y0HgPDCRhttZMIVwRZMBg0CCLwhgIvbbrvNeAnoC1CSCTTo909\/+lNVmMG+e8aMGTJr1qxqoYTPPvvMhDEAEYQsGjRosNp47GL95ptvmusADYDTI488YoZNWAPPyr333mvAStCjgZdixYoVZuEPAgjACR4ZgJjNYTn11FNzAg3CSySjAmwAGJ06dZLXXnvNACobpgnykPcSTvr6669lt912M7kl22yzjQKNmjM7q70pkUCDSQe6X7p0qey+++6y2WabZWUhkyyzzDCuN8rP5muuo92o9UUXo6g56k5\/Ktvsssg1x3PlIJMICtDIbIUAjVtvvdWELAAILOh8bZM3gBeAkAON3SXkcrCY210nQaDBbgzCBwAGvuTvuusus+hnAg3AyDvvvFPlQbFAgX95V2bOwtFHH21CE3gj8BjgEQg28iEIxQBSyN\/gnVdddZW8+OKL5jbrMeH63XffLV9++WXVLhHCGQCiTKDB73hOGCtgCbAAWMnl0QBo\/Otf\/zJgA0DCs4RI8NKQN4KHBr7RGC99AuJs2MqCpCBdrtv4tM3bGgEauLRwE6Lk\/B831x577CFrrLFG0RYZlxuK\/\/3335t4JgKZMGGCyTDObG+\/\/bZ069at2p95d1gWt+tKWDTTQh5Im1KH8ccnen2iFbkXSm++XSeEUDITQrN5M3ifBRpswczcdYJnAXuFbSLRkUWZ8zX48icp87jjjhNACIssuQvkJ7AwW6DxzDPPSP\/+\/c2XO3kY5FmQd4DX4OSTTzb2kxBJ8D6SSY899ljTLx9hhGiwu7fccovJbcgEGoROLr\/8ctlwww3NR1lmLgMfddBFyAXvhw2dAEr4YCMfA+8GY4Q2bDEgiN0z2F5oBFgceuihZlx4GgALfBzizcFG83doJ3QTvM+GTgAaeEb+\/Oc\/VwNr1mNx8cUXVyWq4vEAlJHvARCBfxYUKdAIs4zxXY8daDDRyLDO9CwwoXDBZcvUzkcukwL0TMb2OuusY5KImIC40jKLszApMQC4BotpCjSK4Vby7i10MUoeZauP2CdaowAaloMAjX\/967+\/ZSaABrmcr44GCyleCbZo8rVPwauDDjrIJFuyU4NERfImaIQD+MJnK6kFGuQvkNBJ7geLNwmYgAwaCzhf9ewaAcjY+3jfOeecY+wjYYm11lrLgAzyIeifr3wWbtt4B+MhOZRFOVvjADpyHQAJAA12iNAvHmXsODaWPgiJ4CGhT\/4OmGjRooUBGuSOsAWYMWC38c7Q1l9\/fTN+gAxjpn97H3Uw6NPW0YCf3GeTQnneAh87brwetm9AFB+V1mukQKNyFi12oEEiD+gXZQGRkoQEikX4KBGKQDij0LbffvsZBHzWWWeZR1555RXp0aOHSaoCwQcbCsu7QN3ffvutNG7ceDXEnu29CjQKlUYy7\/Np8fWJ1iiBRk1pNsABL0IwkTT4bravAlAIDeAN\/vjjj80CTsImXl3+xX4G7+N5Fnu2lRLuCCZ3ZtLF+3fddVfzMRbcNRO8j8UcDwneE\/6PPScng0RN+g\/W1iBng10lAA1yUYKNd0En9OIpYesqHmbuI8EV0MG14H2FyIGQFNuAeWewcimg5\/TTTzehoczmuo1P27yNHWhkAgMrcGJ5hE9y7T\/PpmDWjQeAsMpDXJNtX9mq8F100UXG22EbCJ\/YaLYwS5LQbiGTr5h70qbUYbT7RK9PtCYRaITpapzX8fjygYYnONtXv3034IGtqXgyADAADbujJs7xldM3niG8KNj\/TMBDvwo0yuFu8c\/GDjTwMuDO6t69e7XREXPDrRYEDWHDt1upbLyQ+20\/AAjch8HGfm+2Z1GGFtSLV4W4HT8kDOVqKGFmI+mIvd9pbHwN8GXhS\/OJXp9oRX8LpZdQBWEEbX5yABtfbEi9JjlFykGxaQU1Ob5i3xU70CCjmFgbcUJb5x7PBK44CsbYbOZCBm73XoNUjzrqKPOIdf098MADVRXhcvVlk0OzgZLgM66j3UJ4Vcw9+tVbDLeSda\/KNru8fJvjydLa+EfruvzTNm9jBxp8YbCdy+79BmzYxFAyjUnuLKZRt54MbRKpaPRFslIwQYi\/k6SFO5AvF1uSFsDDWHLtUrHjcF0Ji+FXIfemTanDaPaJXp9oRe6F0uvbHA+bE75dd13+hepxUuQWO9CwXgcygN966y2zz5qkIwq4UJq32MZ2LbKt2U9O8g8eExKI2FpFw1tBFTn6JpOa\/ePEE8mQxhMCMKEoTr7jjl1XwmJ5FnZ\/2pRa6V3FAZVtiR6NQo9vDVO2P66TyMn5IvkSM7mVUHAp2\/4LHEbBt6XtWPhMwl238WmbtzUCNArW7gJuxFPBdlnyLGiAFrZ+ATpoeDzYhkUiKB4MtmxRsIXGXnGASGYt\/6QpYQFsKuqWtCl1GPE+0esTrZF4NIo9vlXE2BPywTLz0IJ6yMcNXthsNR3scejnn3++qVVhy3tnPl\/s8fFh8yDf9Zo6Fp7iW+yYIZcvzmPhk2bj0zZvYwEaJF1SR5\/aFvZ44VxKTUneXFu78k0Edpuw7SvXlqzgs9wLQifhMeyLgudcR7vlGJBsz6ZNqcP44xO9PtEaCdAo9vjWCICGPQ6dbaq5gEYpx8eHzYNc12vyWHjqiWDHqbUR57HwCjRK1YZonosFaFA0hpK6lJRl3zXV6HI1DtQppo5GNGTn70WBRk1wuXLv8Gnx9YnWsoFGKce3BoDGYYcdZrwa1PkhD4yPG4pHUefHnmFCbQdCuRtssIE5LIydBWwxpaYEW0gBGpzQSiiYDzA8C+SVYU9zHR+f1GPhOcyN02dp8AmewAcqkbJ5AE81H4Z4PChQxnpCLSYqhlLcC\/4QHre7DfMdC69Ao3L2ljfHAjSCJJGTwWE42bwWIFhO1csse1tZlqhHo9L8j\/v9Pi2+PtFaNtAo5fjWANCg5Db5YXhZqWLJYkgdB0IheAnYHs\/Jo4ANdt1Rn4Kjzm2pbUAFQIPj2ll4KVVO5VCKGlKoK9vx8Uk+Fh7QgCeDyqmUEQeAsSYQ9qa8uS3CaBP+CYFzVgx8JSQOkINngDLuz3csvAKNuK1q\/v5jBxrUn2jXrp2ccsop1UaCAWRHSLaT9SrLEgUaleZ\/3O\/3afH1idbYgUaOA09sjoYFGiS+8zd7xDpf7oRvARokxFM4cObMmaZUOL9nAg1bJwgggsdjwIABOY+PT\/qx8MHQSfBsE3gFaANQcMwEZQwIyXMuDP8eccQRxkyQ10JJ88GDB+c9Fl6BRtxWtUJAAzcihwixrZVmjwO2w7F\/JznKJnJWlhWr3q6hE1ckEc84fFp8faK1bKBhfLzVy2ZXaWCu41uzeDSC5byxJXyBs+vNJnPSJx4OwiwUDcsEGhaMcB8LLqCFLfrZTnVN+rHwuYDGtGnTzImy1FmigjSAipBKtkR+djBSGTrfsfAKNOKxpYX2GptHgxoW1OKnJj718DPPISGcstdee5k6+641BRquSSTa8fi0+PpEayRAgzyNYo5vzQI0ONKc80hoQaAR3HWSD2iQz0FImYbdJPeAPIVsQCPpx8LnAhokv7Zu3docSc\/BcWwsYOsvuwpZWzh2goaniBLjhKvyHQuvQCNaG1psb7EBDTsQEno4sjcTaFhUD1LXZNBixRbt\/boYRctPl3pT2WaXRujHRKHHt8YANM444wwDKqh2TOjA5nhkOz4+6cfCUxeJvAwAUzB0gtRI9CTsRIFGQiO0ww8\/3ISRACicu0LRR7w6FG3Mdyy8Ao3KWqXYgQbkzZ0716BNYo62kQBE7LGQ0uE1zaJQI1TTA4r5fboYxczgCnavsi0RaBQhs8wcjVI8GpzjZOtotGnTxpxKTSOplPCJ3bWCJyR4fHzSj4XHI3HaaacZGimiGDwW\/pFHHjEgYurUqVWeb0DXqaeeanhAI7kWkEKRs3zHwivQKEKhY7g1dqBBfJLtWtnadtttZyp8ulAJLzg+BRoxaJpDXfq0+PpEKypWKL2uz3EWXBZejofP1tJ0LDxnWBFK56eQBu2ffPKJ8YTbEBPP5TsWXoFGIZyN757YgQYonSQd9oyDWtmFwhYuXGGce2JdYvGRWHzPrhuh4inK\/0Shxjnq91aqP5\/o9YnWNAGNOOaGz8fCK9CIQ6MK7zN2oBEsz8tx7XgvcAeyL7xjx47y6quvmkItLjUFGi5JI\/qx+LT4+kSrAo3o50pae3Tdxqdt3sYONPBgNG7cWIYPH26yhcnJeOihh8zpquyTtnvGXVJo15Uwal6lTanD+OMTvT7RqkAjTPP1uuWA6zY+bfM2dqDBmSeER6h0R4ldinRRbIWtSpQnZ9vS+uuv79QMcF0Jo2ZW2pQ6jD8+0esTrcUADXYpUHFSm58coJKoPfHbRQ6kbd7GDjRI3KHcLh4MTjjkhL6JEyca2VIZ7\/TTT3dOzgo0nBNJpANK2yTOxxyfaC0GaESqUBXqTGVbIcbXwGvTJtvYgUY2mQA6aK7Vz0iKWy1qPU+bUofxxyd6faJVgUaY5if7uk+6nDZaYwUanMTHNi22sVpQAcigYijnABA6YU90ri1clZoW6tGoFOdr5r1pm8Tq0VjFAZVtzcyhSrxFZVsJrkfzzliAxsqVK81pfJxcaBvHJpMEygl9weZijkbPTTeVSZ99Fg2HE9CLTxNYv3oToJBlDNEnXfaJVp23ZUwKBx6NBWhQxY6aGV26dDHnmUyfPt0kfdK22morU1iFY+MpSENlt7p16zrAiv87UCnzmOhBg0QGDnRjbDGOQg1WjMytcNcq2woLIMbXq2xjZG6Fu06bbGMBGrfccos5eW\/evHlSu3Ztc4Irh+FQWpcysc4Ai6AyZYIMe80DsJE2pQ6zET7R6xOt+tUbpvnJvu6TLqeN1liAxhVXXCFvvPGG8WTYxrHxf\/7zn+Xyyy93U9tzHQ\/NaFeudHPMEY0qbUodxhaf6PWJVgUaYZqf7Os+6XLaaI0NaLz11lvmMBzbunfvbo79veiii9zTdk5q7NAh97hmzhRp3969cUc0orQpdRhbfKLXJ1oVaIRpfrKv+6TLaaNVgYade+rRSLYVKmL0aZvE+Uj3iVYFGkVMggTe6pMup43W2IAGxxdfcMEFVeo8evRoadq0qakOumptryUHH3ywG6e3ao5GAk1PaUNO2ySuUaCB989h757KtrQ5kYSnVLZJkFL2McYGNG677baCuJJreyuHrr388svSrFkzE3Ip5AjhBQsWyJw5c8yOFxo1OzLLDFPPg\/KzWZvuOilIZkm\/SQ1WCRJMyNxQ2ZYg24Q8orJNiKCyDDMWoPHZZ5+Zo+ELadtvv73UqVOn2q2ULB86dKgp9PXuu+8aYEDZ8nyVRJctWyaHHnqofPPNN0J+CO3tt9+Wbt26Vesbr8qzzz6rQCPAAZ8mMGT7RG8ktCbI2xcJvYUYLgfu8YlWnbcOKFwZQ4gFaJQxHlm0aJG0a9dOhgwZIr169TKVRdu2bSsDBw4UToLN1QjT3HfffVK\/fv0qoPHoo4\/KqFGj5Omnnw4fUoKMaTgxxd2hBqs4fiXp7khkm6D8pUjoTYiAl9xzjzTu0SMhoy1\/mD7JNm20Ogc0KEvO4Wvz58+v8mD06NFD6tWrZ46Uz9Y4eh5gAhAZP358FdAYN26c8WpwRP23335rjqvP9J5U9ZcgY1r+lK3eQ9qUOow\/PtFbNq0J25FVNr1hyuPC9YSEsaJmlRey\/YNpaaPVOaBB0iilyp988skqPR0wYIDMnj07q2di4cKFpgLpjTfeKIRsCLnY0AlbaYNl0PF20H\/Hjh2rz4GEGVOdwOVxIG2TOB83IqE1QSA8EnrLU694n1bPa7z8daT3tOmxc0ADcPDee+9VK\/Y1ZswYU5MjM7dixYoVcswxx5hCYJdddplMnjy5GtDAM8LhbZdeeqmpTkoY5bnnnjM\/TZo0qa5SeYzpls2bG2+JTTJ1RBcjG8bixYvNjiBfmk\/0RkHr+qNHy3qjR6+mHt\/27Svf9O3rlNpEQa9TBGUMpvmWW+Yc3v9+9JHLQy97bGmXbZBB5By2atWqbJ650oFzQGPEiBEybdo0s3vEtmHDhgmeC8IiwQaw4PA2wMlaa61lnnn44Ydl0KBB5gwVQiXBZpND8Wp07dp11SX1aEjz5s1d0cnYx5G2r4XYPRq8ICHu+lTLVu2UN3YqbXpcI0Bj5syZJufiq6++Ws0mXnjhhSaB0zYLHubOnWsOXqNRVXTvvfc2uRvBRrJnsMw5iJddKvvvv7+ceeaZwuFunTp1kpYtW5rH8JR07txZOEl2tfBJgtzDUa\/Ezit1xLUbnKc3QgFHTmvEsoiQVNNV5PRGPcBy+1M7VS4HE\/F82vQ4dqBBQubIkSONcDfZZJPVDlTDA7HOOutUCR8wwuFrxx13nCn4BVjo06eP3H333bLHHnuYM1SeeOIJOeecc1bb7gro4CwVm6MB2KC2BkfTL1261IyDuhr0ycmx1ZrGPt2bgDF9RadtEteIR8M97cg6otTLVu1UQjSxvGGmTY9jBxqEMCi2xdZTdo4U0p555hk57bTTqm7t16+f9O7d2\/w+adIkITkUwJEJFngH22At0MCD0b9\/f3nzzTfNsxtuuKFJBsU7krXFtLAVQnMl73FSqWM0qE7SG5MC+ESrFx4NiFQ7FdNscafbtM3bGgEaFNLKDHuEifTXX3+VDz\/8ULbYYou8hbrC+uH6kiVLZPny5SbhsZAj6ntuuqlM+uyzQrpOxT1OKnWMLmIn6Y1Jk3yi1Rug8YeuaB2NmCaNA92mbd7GDjSuvfZas111xowZ0qBBAwdEGD6ELbfcUj5KeQZ3kAvOKXXMSW\/O0RuukiXf4ROtvgENlW3J08L5B9Mm21iABltROaeExhZUinARtuDMEpvgybXatWvL4MGDy\/ZYRK01CjSi5mgJ\/alHowSmrf5I2gxWGFN8otcnWhVEhmm+29djARo33XSTPP744wVRTrXPIPgo6KGYb1KgETODC+leczQK4VLoPboYhbIosTeobBMrutCBp022sQCNUC46foMCDQcEpEAjEiGkzWCFMcUnen2iVT0aYZrv9vXYgQY7Qahtka1x7sjGG28se+21l3CKqytNgYYDktDQSSRC0MUoEjY62YnK1kmxRDKotMk2dqBx3XXXyfXXX1\/FfHZ+UFiLRt7GL7\/8YsqDX3nllcLhaS40BRoVloImg0YmgLQZrDDG+ESvT7SqRyNM892+HjvQuOuuu8w5JBTc4kySNdZYQ7755hs577zzDGduvfVWcyDaNddc48xODwUaDiitejQiEYIuRpGw0clOVLbuiCXqgrlpk23sQKNnz56yzTbbmN0lwWbLgVN4i3NKCJ24sqVUgYYDE1hzNCIRQtoMVhhTfKLXJ1pd9WjEVTstbbKNHWgcffTR0qxZM3NyarBRRpxqny+++KL5M9U6FWiEmdF4rjur1DHNYmfpjUG8PtHq6mIUg1hNlyrbuDhbWL8xfgulTraxAw22r3KaKmeXtG\/f3pxr8tprrwl\/X2+99WTKlCnmLBP+ZkuHFybm+O5Sj0Z8vC2p54j9kj4ZaJ9o9W3xVdmWZE0ieyjG6K4CjWKltHLlShk7dqxQITTYtttuOyFRlJNb8Xpw4Nm+++5bbPex3K9AIxa2OtOpTwbaJ1oVaDgzxWIZiEu6HHO+ugKNUjVo2bJlsnDhQuHfxo0by2abbVZ17shvv\/1W0Bkkpb672OcUaBTLsWTd75LBiptzPtGqQCNubaps\/67psno0CteH2EMnJH3+8MMPOUfUqlUrp0AGA1WgUbgCJfFO1wxWnDz0iVYFGnFqUuX7dk2XNUejcJ2IHWiceOKJ8txzz+UcEbkZ66+\/fuEjroE7FWjUAJMr+ArXDFacrPCJVgUacWpS5ft2UZdjylfX0Emx6ka4hIJctn333Xfm+Pfhw4fLQQcdJFdffbXUyueDKvaFEdyvQCMCJjrchYsGKy52+USrAo24tMiNfl3W5Yjz1RVoRKVy1M848sgjZf78+Xp6a1RMLbEflydwiSTlfcwnen2iVYFGHLPFnT590uW00Rp76CSXms6bN08OP\/xw4Uj5XXfd1R1t1hwNp2QRx2DSNonz8cgnWhVoxDFb3OnTJ11OG62xAw3yM7788stq2vrTTz\/JPffcI++884689NJL0qhRI3e0WYGGU7Iwg4nYL5m2SaxAYxUHVLbuTd+oRqSyjYqTNd9P7EAjVzIoB6pdcMEF0r1795qnOuSNmqPhiEhiyrRSg+WIfGMYhso2BqY60qXK1hFBlDCM2IEGW1tXrFhRbWh169Y1FUJdbQo0HJBMjHvH1GA5IN+YhqCyjYmxDnSrsnVACCUOIXagwbg4rZVQyQcffCC\/\/\/67NG\/eXP7yl7+Yf11sCjQckEqM1XDUYDkg35iGoLKNibEOdKuydUAIJQ4hdqDx8ccfS9euXc0WV8IltKVLl5p\/KTt+1FFHlTj0+B5ToBEfbwvqOeb6vmqwCpJCIm9S2SZSbAUNWmVbEJucvCl2oMGBaQsWLJCJEyfKFltsYZjw+eefy4033ih33nmnOUyt2IJdy5cvN0mkAJbdd9\/dlDMPa9TvmDRpknBsfcOGDfPerkAjjJs1cF09GpEw2SfjDMN8otcnWlW2kZiDinUSO9AACJx77rnSq1evakT+\/PPPsuOOO5rFf8899yyYAZyVcvDBB8v3339vdqsw2SZMmCAdO3bM2weA55FHHpGnnnrKlBjP1xRoFCyO+G5UoBEJb3UxioSNTnaisnVSLJEMKm2yjR1o7LfffsLOk1NPPbWaAAAMu+yyS0EgIfjg5ZdfLrNmzZKHH37YJJQOGTLEHDWPZ6RevXpZhTxt2jTp16+fuaZAY3UWOafUGjqJxFj59hXoG73OzdvItDZ7Rz7RmzZaYwcaFggMHjxY9thjD9l4443lrbfekptvvlmefPLJokMnAJdjjz1WzjrrLKONr7zyivTo0UMmT55swiiZ7aOPPpL9999fhg4dKgMGDFCgkWUOO6nU6tGIxGw7KdtIKNPFSGUboyJVuOu0yTZ2oEGIBG8GORWZbfTo0SZRtNBGbsb2228v48aNkwMPPNA8tmTJEtlnn31k7Nixcsghh1Trivspc96hQwdThRTAUahHI3NMeGVOOOGEQoeaqPsWL14sTZs2dWrM648eLeuNHr3amL7t21e+6du3rLG6SG9ZBOV52CdaYYNP9PpEq2+yxePPyeZpabEDDcsogAaVQNnqyqKGZ6Jx48ZF8fHTTz+Vtm3byh133CH77ruveZYaHS1atJBsoIWD255\/\/nm57777jAEqBmjgCfGlOYuetWBX2SrorGzLpkw9GirbmJTIgW7TJtvYgcbRRx8tO++8s1x22WVli8\/mdQS3xQJcOCvlgQcekJYtW1a9gxNiDzjgABNmAYhQBv2GG26Q3r17G28I+SG5miaDli2qaDvQEuQl8zNtBiuMET7R6xOtyN0netNGa+xAw+ZozJ49Wxo0aBBmJ0Kv77TTTnL88cdXJXfiKWHLauYpsHgk8GjYRoVS7m3Tpo0cc8wxcsQRRyjQ+IMDaVPqMCXyiV6faNXFKEzzk33dJ11OG62xA41HH31Uzj77bFMFlDyJzJoZFOxac801C54BI0aMkPHjx8uMGTNkk002Ebat1qlTx2yTpRFCYdssYZJgW7hwobRv377gHA0NnRQsksTdmLZJnE8APtGqQCNxU7GoAfuky2mjNXagketQNathxRbsIrmU8AenwtIAMHfddZcBHTQ8HtTsuOiii6op8aJFi6Rdu3YKNLJM7bQpdZj18olen2hVoBGm+cm+7pMup43WWIAG4IHqn126dDFnm6xcuTKnhnPAWimN3Sa\/\/PJLLOelaI5GKRJJzjNpm8Tq0VjFAZVtcuZhsSNV2RbLMXfujwVoXHHFFaa+xUMPPeQOpUWMRIFGEcxK4K1qsBIotAKHrLItkFEJvE1lm0Ch\/TFkBRpZZKdAI7kKXcjI1WAVwqVk3qOyTabcChm1yrYQLrl5jwINBRpebRtD3Gqw3DRGUYxKZRsFF93sQ2XrplwKGVVsQOO2226rStDMNxB2pYSdploIIVHeox6NKLnpXl9qsNyTSVQjUtlGxUn3+lHZuieTQkcUG9CYOnWq7LXXXqHjuOaaayKprxH6oiJuUKBRBLMSeKsarAQKrcAhq2wLZFQCb1PZJlBofww5NqChyaDJUQqfJrCGTpKjl6WM1Cdd9olWnbelzAZ3nlGgkUUW6tFwR0HNSLQEeckC0cWoZNY5\/6DK1nkRlTzAtMlWgYYCDXeTI\/VQtZINlX0wbQYrjCE+0esTrerRCNN8t6\/HAjQo980BaMFDzmADB5vVq1fPueTPTBGpR8MBpc0EGXZIgwaJDBxY1gB9MtA+0aqLUVnTwvmHfdLltNEaC9DIpbGHHHKI7LDDDjJq1CinlVqBhgPiqVUr9yDyVJotZORpm8T5aPaJVgUahWh\/cu\/xSYpA6XkAACAASURBVJfTRqsCDQ2duBc6ISejQ4fcFnHmTJH27Uu2mGmbxAo0VnFAZVvytHD+QZWt8yLKOUAFGgo03AMayEQ9GpFYFZ+Ms28ejXvuWSI9ejSORE+S0IlPupw2WmsUaIwZM0YaNWok3bt3d1qvNXTigHg0RyMSIaTNYIUxxQd6Y8qRDmNtxa\/7IFvL5LTRWmNA49dff5XFixeb01ybNm0qa6+9dsUVN9cAFGg4IhrCJ4RRbCNcQtikzJa2SayhE39CJzHi7zJnVfyP67yNn8dxvSF2oAGwuPHGG+Xaa6+tRsMRRxwh\/fv3l4033jgu2kruV4FGyayL7sEYLaoarOjE5FpPaZdtjBFF10S52njSLtsgwWmjNXagceedd8rll18uAIu9995batWqJW+88YZMnz5dtttuO7n77rulbt26Tim5Ag0HxBGjRU3bJFaPhh8ejZhzpB2Y9PmHoPPWeRHlHGDsQIMtrW3atJFB1D8ItLffflu6desmjz32mGy77bZOcVCBRoXFEbNFVYNVYfnG+Pq0yzZG\/B2jVKLpOu2yVY9GGXqy0047yRVXXCGHH354tV7+85\/\/yNZbby3jxo2TAw88sIw3RP+oAo3oeVp0jzFaVDVYRUsjMQ+kXbYxRhSdl3HaZatAowwV7NGjh6y77romTyMYInn++eflhBNOkAceeGC1CqJlvC6SRxVoRMLG8jqJ0aKqwSpPNC4\/7YNsddeJyxoYzdjSpsexh05eeukl6dmzp2yyySay3377mR0nc+bMkeeee86EVCZNmiS1a9eORjoR9aJAIyJGlttNTBY1bZM4H5t9ohU++ESv1tEo18C4+3za9Dh2oIEoOTL+mmuuMf\/S6tevL3g6+vTp4+S5Jwo0HJuAenpryQJJm8EKY4RP9PpEq28gMm2yjR1ovPjii7LRRhuZhM+VK1fKL7\/8YmporFixQh588EHp3Llz1poaH3\/8sbz88svSrFkzad26tay55po5bczcuXNl\/vz55t5dd921qr+ff\/5Z8KgEG+\/ec88989orBRph5jzZ19M2idWjsYoDKttkz03V5f9yIG16HBvQoDgXRbrOP\/982WuvveToo4+upkPvvfeenH322fLoo49KixYtql279dZbZejQoWb767vvvmuAwcSJE7MCEqqNUqNjww03NCDmxx9\/lKeeekoAC3ZnS7BzQjfPPvusAo0AB9Km1GGm1id6faI1jQZaF14FkWH2LAnXYwMa1Mz4\/PPP8\/KAEMoLL7xgkkVtW7RokbRr106GDBkivXr1kk8\/\/VTatm0rAwcOlBNPPLFaf3gsdtxxR+nbt6\/07t3b1OjYf\/\/9zQ+1OwAxnBT79NNPFyUL9WgUxa7E3ezT4usTrQo0EjcVixqwT7qcNlpjAxokfC5fvlyGDx9udpUceuih1ZSqXr165sj4Bg0aVPs74RSAA6EQW6acfA7uv+OOO6rdizAGDx5s8j822GADU968a9euss0228jo0aPN1lm8Gozh22+\/lcaNG0udOnVClVuBRiiLEn1D2iaxfvXqV2+iJ2SBg9d5WyCjHLwtNqBhaX3ttdcMCGjevHlB5AMQHnroIXnyySer7h8wYIDMnj07r2eCcMvrr78ujzzyiNxzzz1mR8tFF10kU6ZMqeoHDwr9d+zYMe9YFGgUJKrE3qQGK7GiCx24yjaURYm9QWWbWNFJ7EAjyJrTTz\/dAI5LLrkkJ8cAB+RvUKLcNvIwpk6dmje3Ai8IiadLly41BcKOO+444xn54Ycf5NJLLzW5G4RR2FbLT5MmTXKOAaCR2QjbUPcjjY18GnJXfGk+0esTreivT\/T6RKtvsl22bJm0atUqNSa5RoEG5cgJl7Dg52ojRoyQadOmmVobtg0bNkwWLlwo48ePr\/YY1UX5CRYCu+CCC+S+++6TBQsWrLZTxSaH4tUgxJKrqUcjNfqdlRD9MkqvfFW2Kts0cCBteuwc0Jg8ebJcfPHFwpZVmyTavXt3cyAbHopgo6roueeeWw1UAFL69esnzzzzjMyYMUM6depUVXkUTwnbaSdMmJA3fKJAIw1TNTcNaZvE+aTlE63wwSd6faJVZZtsm1yjQOPUU0+VrbbaKm\/o5KuvvjL5FYQ+8E5QqpzCXpzyuscee5iTX5944gk555xz5LfffjPuJUIap512mvAsYRmSUNlpcuaZZxoQcv\/995uQysiRI01dDfps2LChejT+4IAarGRPYgUaqzjgky77RKsCjWTbqNiBBuEKdp1kNkIe5F2wG4UkzWDDGwFwsA0PBdtXaZQsJzkUwAFYuOWWW+Sqq66qupcckLFjx8r2229vcj369+8vb775prlOrQ3CJnhH8jX1aCRbqcNG75OB9olWXYzCND\/Z133S5bTRGjvQIImScAVbVG2DiYRHKEnOrpT1119\/tRlAsa8PP\/xQtthii6yFuoIPkDhDDgehFpI8M7ewLlmyxHg5SHgM5nPkmnYKNJJtkMJGn7ZJrB4N9WiE6Xwaruu8Ta4UYwcaN910k1x99dUmAbRbt27CNlTqWtDYYUI4pZDaFjXJYgUaNcntmn+XGqya53lNvVFlW1Ocrvn3qGxrnudRvTF2oMFAARkcE09YA2WhcieVPvNtMY2KwFL6UaBRCteS84warOTIqtiRqmyL5Vhy7lfZJkdWmSOtEaDBS9m2yvbUY489Vtiu6nJToOGydMofmxqs8nnoag8qW1clU\/64VLbl87BSPcQCNDjU7NVXX61GE8mfbCulnXTSSaakOI0tq7bUeKWYkPleBRquSCKecajBioevLvSqsnVBCvGMQWUbD19rotdYgAZ5GZxZUkijbsY666xTyK01do8CjRpjdUVepAardLbPmiXSvn3pz8f9pMo2bg5Xrn+VbeV4X+6bYwEahQ4KL0ft2rULvb3G7lOgUWOsrsiL1GAVz\/bBg0UGDVr1HP8fOLD4fuJ+QmUbN4cr17\/KtnK8L\/fNNQI02Kr62WefycqVK6vGu2LFCjn44IPl4YcfNjUvXGoKNFySRvRjUYNVHE8zQYZ92kWwobItTrZJultlmyRpVR9r7EBj3rx5cvjhh2flEIW6OAhNQyeVVSDnJ3DE\/nrn6Y1QHaKgtVat3AMKfDtEOOrSu4qC3tLfXrNP+kQrnPWJ3rTRGjvQGDRokDktlaRPTlE95phjZNtttzUVOnv27GnKi7vW1KPhiERi8tenbRLnk1a5tILxOnTI\/YaZM93K2SiXXkc0v6Bh+ESrAo2CVMLZm2IHGgcccIAcf\/zx5jwSzi7ZaKONTKEuKoIeffTRMn\/+fN11UmH1cNJgxeivd5LemHQgClrVoxGTcMrsNgrZljmEGn3cJ3rTRmvsQOOwww6Tdu3ayXnnnSfXXXedOaPk9ttvF3I0WrRoIew62X333WtUYcNeph6NMA7VwPUYV7e0TeI4PRr0HSPmi1yRVLaRs9SZDlW2zoii6IHEDjSuvfZaU3acgl0cakbBLk5R\/emnn\/4vi32QOc6d80xcago0KiyNmP31arCKl29MUaziBxLyhMo2cpY606HK1hlRFD2Q2IHGl19+KRdeeKGsscYacvPNN5vD1ebMmWMGSliFaqGuNQUaDkhEPRqRCCEq46xAIxJxRNpJVLKNdFAxduYTvWmjNXagkU3vCJ\/UqlVLdtlllxjVsvSuFWiUzrvInozRX5+2Sayhk1UcUNlGNgOd60hl65xICh5QjQCNzz\/\/XB555BH5+OOPzY4T6mZQrMu13AzLNQUaBetPvDfG9BmtBqs4scXoXCpuIAXcrbItgEkJvUVlm1DB\/V+eV+xAA5DRtWtXWbp0qeHSEUccITvssIMMHTpUBgwYIKeccopz3FOg4ZhItI5GyQIp1zjHnC5TMl25HiyX3sgHFGOHPtEKG32iN220xg40qJ1BHY077rhDZsyYIYsWLTKJoQANdp+88sorZsurS02BhkvSiH4saZvEcYdO1KMRvQ5G0aNPeqxAIwqNqVwfsQONTp06mcJcp556qinSBdAYNWqUKUm+7777ysSJE6VDvopAFeCNAo0KML0GX+mTgY6C1hjTZSKXehT0Rj6omDr0iVYFGjEpUQ11GzvQoPx406ZNZcyYMdWABgmhRx55pNx5552y99571xC5hb1GgUZhfErqXT4Z6KhojSldJnIVioreyAcWQ4c+0apAIwYFqsEuYwca1NAgTHL22WfLp59+Kt9\/\/705+4Strh9++KHMnj1bGjRoUIMkh79KgUY4j5J8h08GOmpaI06XiVyNoqY38gFG2KFPtCrQiFBxKtBV7EDj999\/lyFDhpgcjWCjeNcNN9wgbdq0qQDZ+V+pQMM5kUQ6IJ8MtE+06mIU6TRxrjOfdDlttMYONKy2wri3335bli1bJo0bN5bddttN1l13XeeUmQEp0HBSLJENKm2TOB9jfKJVgUZkU8TJjnzS5bTRGhvQWL58uTz99NOyZMkSk4NB7YyoGn2\/9NJLZssstTg222yznF1Tu4N8kLp165rk0\/XXXz90GAo0QlmU6BvSNokVaKzigMo20VMz7+BVtsmVbWxAgzyMefPmVXGmVatWMmnSpLJPasUjcvDBB5tcj0aNGpm91RMmTJCOHTuuJoXnn3\/enBpL22STTYSaHux4oZZHvqZAI7kKXcjI1WAVwqVk3qOyTabcChm1yrYQLrl5TyxAgxBJt27dzHHwbdu2lQcffFDGjRsnV1xxhRx33HFlceLyyy+XWbNmycMPPyzrrLOOyf+YMmWKOXa+Xr161fpmWy2JpsOGDTO1Oi6++GJz3xNPPGFKoOdqCjTKEpHzD6vBcl5EJQ9QZVsy65x\/UGXrvIhyDjAWoMFx8HfddVfV4WkrV640SZ9dunQxJ7aW0\/bbbz9zAuxZZ51luqHgFwe1ZR43zzt79+4tJ598suy5557mXkDJAw88YMIuderUUaDxBwdcn8BR73Rwnd5y5kfmsz7RCu0+0esTrSrbKK1CzfcVC9DAc0FexPTp06soOumkk0z4Yvjw4SVTSW4GuR54Rw488EDTDzkg++yzj4wdO1YOOeSQrH3jwXj11VdNiAUvyxlnnJF3DOrRKFlEkT4YV+0Gnwy0T7TqYhTp9HOuM590OW20xgY03nrrLZk6dWqVslIZdOONNy4LaFCHg1AMW2VJ7KStWLFCWrRoYYqBcaZKtkbNDjweCI\/8DEqgh3k0Mvs58cQTq\/I9nJuBZQ5o8eLFpqiaS2306PVl9Oj1VhtS377fSt++35Q1VBfpLYugPA\/7RCts8Ilen2j1TbbkIpLXmJYWG9B4\/fXXjQfBtvPPP18aNmwoAwcOrMa7Ys45gfkcLT9y5Eg56qijTD\/ffPON7LrrriYk0rJly6q+CZ1Qw4PdJrYBfPr37y\/33nuvtG7dOqcM1aNRefWO83yNtH0t5JOWT7SqR6Py8zbOEfiky2mjNTagcdtttxWkcyRnFrLl1Ha20047yfHHHy\/9+vUzfyLfgqTP+fPnV9vRYr0fgAuAiDVCnL2SL8zCfQo0ChJdbDfFfWJo2iaxAo1VHFDZxjYtK96xyrbiIih5ALEADbaVsvOkkEay5pprrlnIreYewh7jx483J8GS89GnTx8TBmHrLI0Qyo477ij777+\/KXW+9tpry4UXXmjAzPXXXy\/333+\/OU22SZMm6tH4gwMuTmD1aBQ8JfLe6KJso6Esey8+0esTreqtinPWxN93LEAjzmH\/\/PPPZjcJYIHWvHlzs8MF0EHD49GrVy+T9AngYXfKjz\/+aK7Vr1\/fAJVcSaN23OrRiFOChfUd54mhPhlon2jVxaiwuZXUu3zS5bTRmjigYScJu01++eUXAzTyNXaqfPLJJ+aWZs2aFeQ9UaDhhinSXSflyyFtBiuMIz7R6xOtCiLDNN\/t64kFGnGyVYFGnNwtvm8AR0YOcfGdBJ7wyUD7RKsuRmVNC+cf9kmX00arAo0s00uBhhs2Rz0a5cshbQYrjCM+0esTrQoiwzTf7esKNBRoOFlNUXM0ojEcuhhFw0cXe1HZuiiVaMaUNtkq0FCg4STQ0F0narBK4UDaDHQ+HvhEq3o0SpkN7jyjQEOBhnNAQ+toRGcgdDGKjpeu9aSydU0i0Y0nbbJVoKFAwzmggUjUoxGN0UqbwQrjik\/0+kSrejTCNN\/t6wo0FGg4BzTUoxGd0dDFKDpeutaTytY1iUQ3nrTJVoGGAg3ngIZ6NNRglcqBtBlozdFYxQGVbamzovLPKdBQoOEk0NBdJ9EYB5+Ms7rXo9EZV3vxSZfTRqsCDQUaTgINxKJ1NMo3+WkzWGEc8Ylen2hVEBmm+W5fV6ChQMNZoGFFQ85G+\/bRTSSfDLRPtOpiFN0ccbEnn3Q5bbQq0FCg4TzQiNropW0Saxxf4\/hRzxEX+9N566JUChuTAg0FGgo0CpsribzLJ+OsHo1EqmjBg\/ZJl9NGqwINBRoKNAo2dcm7MW0GK0wCPtHrE60KIsM03+3rCjQUaCjQcHuOljU6XYzKYp\/TD6tsnRZPWYNLm2wVaCjQUKBRlklw++G0GawwbvtEr0+0qkcjTPPdvq5AQ4GGAg2352hZo9PFqCz2Of2wytZp8ZQ1uLTJVoGGAg3ngYZuby3dZqXNYIVxwid6faJVPRphmu\/2dQUaCjScBRpasKt846GLUfk8dLUHla2rkil\/XGmTrQINBRpOAg0tQV6+sfLtK9A3etO2GIVpvE\/0po1WBRoKNJwEGnpMfJjZLex62gxWGNU+0esTrQoiwzTf7esKNBRoOAc09Jj46IyGLkbR8dK1nlS2rkkkuvGkTbbOAo2PP\/5YXn75ZWnWrJm0bt1a1lxzzZxS\/PDDD+WNN96QX375RVq1aiU77bSTuffnn3+Wl156qdpza6+9tuy55555NWLLLbeUjz76KDqtcbwnF5VaPRrRKI2Lso2Gsuy9+ESvT7SqRyPOWRN\/304CjVtvvVWGDh0q2223nbz77rsGGEycOFEACZntwQcflL59+0rz5s1l2bJlsnTpUvNsz5495e2335Zu3bpVe6Rp06by7LPPKtAIcMBFg6U5GtFMfhdlGw1lCjRUtnFqUmX7TptsnQMaixYtknbt2smQIUOkV69e8umnn0rbtm1l4MCBcuKJJ64m\/f322086duwogwYNkt9++0369esnr776qsyaNUsef\/xxGTVqlDz99NNFaY16NIpiV2w3666T8lmbNoMVxhGf6PWJVvVohGm+29edAxrWQzF\/\/vwqD0aPHj2kXr16cscdd1Tj5tdffy277babTJ48WXbffXdz7b777pMLLrjAhEz4P16N4cOHy7fffiuNGzeWOnXqhEpEgUYoi2r0Bq2jUTq7dTEqnXeuP6mydV1CpY8vbbJ1DmiMHj1aHnroIXnyySerpDRgwACZPXt2qGfihx9+kOOPP17WXXdduf322+Wiiy6SKVOmVPVTv359oX88IPkaQCOz4U054YQTStcch59cvHixEFLypflEr0+0or8+0esTrb7JljQA8g3T0pwDGoCD9957T6ZPn17F4zFjxsjUqVPz5lY8\/\/zzcskll5gE0AkTJhghkbsB+Lj00kvlxx9\/NGGU5557zvw0adIkpwzVo5EW9c5OR9q+FvJJyyda4YNP9PpEq8o22TbZOaAxYsQImTZtmsyZM6eKs8OGDZOFCxfK+PHjV+M2eRkkfxJWweNw3nnnScOGDbNKxSaH4tXo2rWrAo0\/OKAGK9mTWIHGKg74pMs+0apAI9k2yjmgQb7FxRdfLHPnzjUhEFr37t1l7733Nh6KzMbfXnzxxSovhr2OZ+Pmm2+WTp06ScuWLc2f8ZR07tzZ3JsvfKIejWQrddjofTLQPtGqi1GY5if7uk+6nDZanQMaX331lbRp00aOO+44k9RJSKRPnz5y9913yx577GHqZTzxxBNyzjnnmDAJyaBnnnnmasBhl112Mc8tWLBA7r\/\/frPtdeTIkSZJlD5zeT2Yigo0km2QwkaftkmsHg31aITpfBqu67xNrhSdAxqw8plnnpHTTjutiqtsWe3du7f5fdKkSUJyKIDj9ddfl1NPPTUr96mVQV5G\/\/795c033zT3bLjhhiYZFO9IvqZAI7kKXcjI1WAVwqVk3qOyTabcChm1yrYQLrl5j5NAA1b9+uuvQsXPLbbYImuhrmLYuWTJElm+fLnZWVG3bt3QRxVohLIo0TeowUq0+PIOXmWrsk0DB9Kmx84CjUoqiwKNSnI\/\/nenbRLn45hPtMIHn+j1iVaVbfx2Mc43KNDIwl0FGnGqXOX79slA+0SrLkaVn1txjsAnXU4brQo0FGh49RWoi1GcS0Hl+06bgVZv1SoOqGwrP79KHYECDQUaCjRKnT0JeM4n46wgMgEKWcYQfdLltNGqQEOBhgKNMoyf64+mzWCF8dsnen2iVUFkmOa7fV2BhgINBRpuz9GyRqeLUVnsc\/phla3T4ilrcGmTrQINBRoKNMoyCW4\/nDaDFcZtn+j1iVb1aIRpvtvXFWgo0FCg4fYcLWt0uhiVxT6nH1bZOi2esgaXNtkq0FCgoUCjLJPg9sNpM1hh3PaJXp9oVY9GmOa7fV2BhgINBRpuz9GyRqeLUVnsc\/phla3T4ilrcGmTrQINBRoKNMoyCW4\/nDaDFcZtn+j1iVb1aIRpvtvXFWgo0FCg4fYcLWt0uhiVxT6nH1bZOi2esgaXNtkq0FCgoUCjLJPg9sNpM1hh3PaJXp9oVY9GmOa7fV2BhgIN54HGrFki7dtHN5F8MtA+0aqLUXRzxMWefNLltNGqQEOBhrNAY\/BgkUGDVgmI\/w8cWL4JTNskzscRn2hVoFH+3HC5B590OW20KtBQoOEk0MgEGVZMUYCNtE1iBRqrOKCydRkqlDc2lW15\/Kvk0wo0FGg4CTRq1co9LVauLG\/KqMEqj38uP62ydVk65Y1NZVse\/yr5tAINBRrOAQ1yMjp0yD0tZs4sL2dDDVYlTU6871bZxsvfSvausq0k98t7twINBRrOAQ1Eoh6N8ia2fdon46w5GtHojKu9+KTLaaNVgYYCDSeBhuZoRGPu02awwrjiE70+0aogMkzz3b6uQEOBhpNAA7HorpPyjYcuRuXz0NUeVLauSqb8caVNtokEGsuXL5eXXnpJli5dKrvvvrtsttlmOSX75ZdfyiuvvCJffPGFNG\/eXNq1aye1a9fOqwlbbrmlfPTRR+VrS0J6cF2ptY5G6YrkumxLpyz7kz7R6xOt6tGIeqbUbH+JAxrLli2Tgw8+WL7\/\/ntp1KiR+RqfMGGCdOzYcTXOvf\/++3LkkUfKWmutJeuuu665t0uXLjJmzBgFGgEODB48WAZGUaCiZnW35Lf5RK9PtP7XC+aPLvtEq8q2ZHPnxIOJAxqXX365zJo1Sx5++GFZZ511ZMiQITJlyhR57bXXpF69etWYeumll8q8efNk6tSpsvbaa8s999wjl1xyiTz33HPSpEmTnALwzaOh9DoxF2MZhMo2FrY60anK1gkxxDKItMk2cUBjv\/32k2OPPVbOOussI2DCIj169JDJkyebMEqwHXLIIdK1a9eqez\/++GPj+bjhhhukc+fOCjT+4EDalDps5vtEr0+0Inef6PWJVpVtmFVz+3qigAa5Gdtvv72MGzdODjzwQMPZJUuWyD777CNjx44VgEWu9p\/\/\/EeGDRtmvB\/kd9SvX1+BhgINt2dnBKPTxSgCJjrahcrWUcFEMKy0yTZRQOPTTz+Vtm3byh133CH77ruvEeeKFSukRYsWMnr0aOO9yNbIzRg0aJAJmYwcOVKOOuqovKrQs2dPA0a0KQeUA8oB5YByoKY5sOeee8qkSZNq+rWxvS9RQINE0F122aUaWPjmm29k1113lQceeEBatmy5GqOmT58uf\/\/73w0wAWyAFLUpB5QDygHlgHJAOVAzHEgU0IAlO+20kxx\/\/PHSr18\/wyE8D3gg5s+fbxI+gw1EOGDAABk1apQcccQRNcNRfYtyQDmgHFAOKAeUA1UcSBzQGDFihIwfP15mzJghm2yyifTp00fq1KlT5WYihLLjjjvK\/vvvLwcccIA0btxY\/va3v1UT+VZbbSUbbLCBqoFyQDmgHFAOKAeUAzFzIHFA4+eff5bevXubfAsaRbjuuusuAzqsx6NXr17y17\/+1YRZsjX1cMSsVdq9ckA5oBxQDigH\/uBA4oCGlRy7TX755RcDNLQpB5QDygHlgHJAOeAmBxILNNxkp45KOaAcUA4oB5QDyoEgBxRoqD4oB5QDygHlgHJAORAbBxRoxMZa7Vg5oBxQDigHlAPKAQUaqgPKAeWAckA5oBxQDsTGAQUasbFWO1YOKAeUA8oB5YBywFugwdknb731lrz77rum7gY\/tWrVyqsRL774oikK1rp160RoDofIvfzyy9KsWTMz5jXXXDPvuL\/\/\/nt58MEH5ZhjjpG6detW3fvll1+aw+u++OILs8unXbt2Urt2bad4EBWtlqjvvvvO1GahGFzDhg2dopXBREXv77\/\/LnPnzjWnH++xxx7SqlWr1NL622+\/mQJ\/6HP79u1l\/fXXd47WzAEVY6fY+s88\/fDDD812fw6gbNCggdM0FqPH0PXGG2+Y3YboKcUbbUuCjSp23uajNyl2yo7TW6BxxhlnyFNPPSU77LCDvPPOO6Y2h602mm1mosic\/HrKKafIeeed5\/TkZXC33nqrDB06VLbbbjsDpqidP3HixNWqpwYJ+cc\/\/mFOtg1WWX3\/\/fflyCOPlLXWWkvWXXdd4dyYLl26yJgxY5zhQVS0BgmiENwjjzxidMS1svVR0csiZqvqbr311jJv3jxzDhDnAbnSoqL1o48+kr\/85S+GrHXWWUc+\/\/xzufLKK83Jzy63Qu0UxzOceOKJRobWpgE2nn766bxzvpK0FyNbPoD69u1rPnSgdenSpca+ob9JsFHF2uR89CbFTgXH6SXQYAFhcx4M\/QAADpNJREFUIXn44YfNabBPPPGEARrZzkvhxNgTTjhB5syZY\/hGlVHXgcaiRYuM12HIkCFC8TJ7GN3AgQONMcps9913nynTjvGlBYHGpZdeaozX1KlTjcG655575JJLLjEF05o0aVJJO2XeHSWtlphp06ZVgU7XgEaU9FLobvjw4fLYY48ZWdp5gBds4403TpVsr7jiCvO1j66vscYawu8PPfSQzJ49u5r3ruJEBwZQjJ2yC9OsWbNk8803N4vvwQcfLNdcc40cdthhLpFV0rzFO8OHHudV4Znio\/DVV18V6L3sssuctlGl2Kl89Fpvsst2KlPhvAQaLJ4YbE6Bpdnj5y+44AJTUTTYcC0DQGh8AYGgXQca1ugEAQNfbvXq1auiOUjj22+\/LR988IFxnxMuCD53yCGHmFNxzzrrLPMIrk4mPJ6Pzp07V9yARUkrxPDlS\/l6vpY4J8c1oBElvYDO+vXrG1kSNsPN\/u9\/\/9uEFMLCbDUh+Chp5cOC0B+AmXbdddcZDx9hI44wcLEVY6ewTe+9957cfvvtVaTsvvvuwvwdPHiwc+QVI9uvv\/5adtttN5k8ebJAEw3AiL0mFHbSSSc5baMYb5T0NmrUyHk7pUBDRLp3725OegUJ2xZEkLlmZadOneTQQw91Hmhw3gtfa08++WQVKSyafL3hSs3VHn30UTn77LOzHlDHM7jahw0bJlOmTDETnEWq0i1KWgGchIk6dOgghx9+uAEcrgGNKOnde++9ZYsttjC5Sj\/++KO0adNGTjvtNEO3Cy1KWgHRRx99tIntk3Pz7LPPGiBJKNTVVqqdgh47lwFWnG7tWitVttDxww8\/mIM1CeUGgZWrNopxRUlvEuyUAg0RwcCyoHB8vG188XPYGgtp0oHGRRddZL5upk+fXkUKORUYHQxsKUCD3AzcloRMiOETy3ehRUkrYYTnn3\/efC0tXrzYSaARFb0kDpIATeOgQtztEyZMMEDUFXAVFa3QaN3MzHHCQgBlQgqEFlxtpdgp8hdY1P75z3+a\/BM8HS62UmXL\/CR0i\/6ir8HkZVdtFPyPkt4k2CkFGiLGK0GSJHkJtuFi5AsiWw6DvScpHg0WDgyrzSth\/ACohQsXmpNviwUaABZA2b777mvAhkvJkVHRSoY3p\/0ee+yx0qJFC7MzgZACuTsHHnhgzgP6atqIR0Uvce5tt9222mILzew8cQVIRkUr4U8OWDzooIOq5rx1ZQOsXD0vqVg7RQj05JNPNrlU5GeRp+VqK1a26CvhTMLd5MwRvg7uBnPZRlkwX4xNzkVvUuyUAg0ROf\/8802M6\/777zf8wBW38847y5133mm8HUn3aBDLvPjii038GfciDRAFbWRuFwM0yNnAxezqibdR0Yo+8KVgGzrBVy\/hBLb7HnHEEU7Y7KjohRj0gTCR9eJZoEH+Qrdu3SpOb1S0WrqCu0xsgvTYsWNNHoOLrRg7RfiLXTXkK\/Tv39+JHJt8PC1WttgtygtkejF4h+s2ijFGRW9S7JQCDRF55plnTCyaLze+VgkrsJsCtxwoGQBCAlJm\/DYpHo2vvvrKLJDHHXecSZiCLpLh7r77bvPFyl50dhicc8451ba+ZcvR4Cu\/cePGZrdNsOGC3mCDDSpun6OkNUgM3h9qLbgSRrBji5Je9P7mm2+Wa6+91uQssehitDHobI2sdIuSVkAV+gqoYocN3iri+\/\/6179ks802qzSpWd9fjJ3Cg0Hu1C233FItuZUwEXk4rrViZEuYhGTQM8880ySiBxueKoCiyzaK8UZJL7umbHPVTinQ+IMDtmYEv5LUSEjBejNYVIn3kVAZbAANdmCce+65rs3b1cZjjZS9wHYwwgDBLwAAR9D9mAk0iPcykbM1lzwcUdCaSaPdRuoa0GCcUdFLAihfv2yjtPOAeQH4dqVFRStf\/FdffbUB3ba5pMO5+F2onSLMQj2gzIY3Luipc0Wuxejx66+\/LqeeemrWoVOigLo+rtuoqOglx65p06ZV5Lpsp4Iy8XJ7q2UAW\/pI+ttmm22c3UtfjmH49ddfTZVAvmiI26a5+UQrcoySXra0fvPNN0ZPXNjWmqmnUdIKndSLwYvhetVMH+xUlLJNgn3zjV4rE6+BRhIUU8eoHFAOKAeUA8qBJHNAgUaSpadjVw4oB5QDygHlgOMcUKDhuIB0eOnmAFsSb7zxRuFfwhYUy2rbtq3stddeoYf8Wc5Q+ZEDptgZw\/Y\/ci9sJddyuEdyKJVgyW2wBw7a8s\/sZmKrs23s0mGnE5UbKevPOK666qqs4QkSrakrQKIyW2wzG9dIZi52pw8JzpRTj7o2BlthyROghL825YByoHgOKNAonmf6hHIgEg4sWLDAJLJtuOGGJgGTHQIkn5LUd+GFFxYMFtgZRI0IFm62RLKQ33bbbWWP0SYiBhNiSSCm2B0teN4NO1XY5QT42GijjUySKYlr2U5ItVtL2eHCYX88w7kVNvmapGy2atJHoQ2gY7fqRp3MSlL0PvvsY3Z0MF5tygHlQHEcUKBRHL\/0buVAZBzgLAu2HHPYF4szDY8BRcM4cffNN99czauxcuXKvJ6OKIEGOzQojhTcnYGXw3oMgoW92CrLNll2LlHwLF\/LBBrwAR7YkvlBoBFGr30PXiG2qHPIlj10qlRBUWo\/sw9oY3cOXhPr3Sm1f31OOeAbBxRo+CZxpdcZDnAEOOfP2PotdmAUCqO+A9uoP\/nkE\/NlTzlpqh+yIFNcDjd+69atzSMcBEidAeq+ZAINQhgswGzZ5hA8tsNRc4H34m3ghE+qvmbbbUIIhpBMsJQ1Z8BQh4IDyqi1AcCgUVGX8ALgCHoAI9SpIMQCsOC0VLweVOTF40BYBY8G4+C4cN7FNcrk46Hh7CEy9AEfa621lgE8mbVcgoJknPDBhozYDgmflixZYk5p3nrrrQ1v8CLh7WHHGbRcfvnlZsfZd999Z7a441VhJxqhG2puUC+GZk9DtV4YZ5RIB6IcSAAHFGgkQEg6xHRygGJZ1DjYYYcdzIFfHH7FYsvCZ1swVEHZZRb5m266yWxbtse54wEARHBIYBBo4OpnQbdhGBZTysgDMFiIKc1NWWfyQhhLtgbIoLImpbrZGsoYKR\/NFlGeYXy0YDlzCt4xDsIh1KghpMH9ACsWeLw4NBZtTk4lRAM9jB86KJ3N\/YQpCC1xejLl9IOndwbHyr3wgKqRtqATv\/N36OUHcMHvhKkALfyfcQCUeIctiQ3wYIycFUJhJFtjZMWKFcZTQ9GoYkI66dRcpUo5UBwHFGgUxy+9WzkQGQdY6Fno7rrrLlm6dGlVvyRB8nXOmTIWaPA7ORg0islRPI7ky9NPP90ssplAg0Wa0AZhCVvsiMO2+Hn88cdN7RgaZfdZXDMLAdnBEJIgdAKoAThwui+5GdTeIFfj3nvvNQsyJcttiCUINAhlMG5AiT0VlhLn119\/vQEajDNb6GS99dYzR2sDRABIeG\/I5QAkZDYb4iER1CaXwhP6wENBGASvCaAK4EU5fk7AJGkVLwkADg8KXh3u5188H\/AJD4n19sBzqovao+YjUwTtSDmQcg4o0Ei5gJU89zlAHsIHH3xgQAVufhZOvrxZ6AidsKAThmA3im3s7iAEgXs\/E2jYM3y4N1hOHFCCZyK4WH\/22WcmAZUQC1\/+mc0CHYACVToBHPRBLgmLP1\/4eEjwRliwEgQalPrGOzBv3jxZZ511TPd4J1js8wENPBMAA9sAXZx3ke2sHgsi5s+fX1WYDp4AbAjZ0DiRF8ATTGCFhwAkxm5zTPDAkFiLV4Ufe1YQfUArACTfCcjua5uOUDlQ8xxQoFHzPNc3KgdMTgKJlSyoLHjBZkMe48aNMztRABrkZ9gjsTNPI80GNDjHBq8BORh4JWgs7izynPOT2QjdWC9H8FowZMDpk4yF7ac0Fn3OWuD0U7wddgEOAg1CQzwHkMI7QWMrLwt8PqCRueskH9Cwu2MygUawDws0XnvttaqdMEGgYQEQOSEAKbxGhLGsR8TSCx02nKJqrBxQDhTGAQUahfFJ71IORMoBdjaQoMiBUACOYCMZtGfPnubgLxIuWdxZTEleDC7UnKpLAmgm0CAMgweEXA5qYNAPoRUAAgdvBb0LfKHPmDHD9NOoUaOsNFpPCBeDng+7ePN3vCS2rkYQaAAmSAzFU0OogmY9EFEBDcbEuzNDJ4UCDXIu2FWCLEgqpQEmOIgwSG+2Y9sjVQrtTDmQUg4o0EipYJUs9zlAoS0AAS55vvBx07O7gZABtRtY7NgBAdAglEL4AoBC7gWeCb688SbkSgYlD4HFk74Ij9A3Xo3DDjvM5HawAwUgwn0zZ86sdupnkHuMkbHSgl4Dwi423GLBDPcEgQbvxmtDGINcELwZhCoAQxZo2JNHAUGAkWx1NPJ5NOwR6cE8kMw+wjwa7Jr56aefhMMHSfqcOHGiCacA+iwAYww2L8Z97dIRKgfc4YACDXdkoSPxjAMcf81uB76ag42kQ8ISHN1ucyRImmTRo3F6I94JQAQtE2iQPMlCSWM7LDtHqM1BPodN\/rTvA6gQWslX+8IWFmP3iO3XPs9WVHaMEDohUZLGLhG25tq\/kdQZzK0AWLGV1AINW+yLZ+fOnWvCPcWETmy+CImq9oTiXEAjeGJxMHQCcGObL8DONnI6SMKl2Z0tmbkynqmskqscKIkDCjRKYps+pByIjgMAA0p903bccces21tZvJs1ayZ4EdhZUU7RKKpoAg440Ze8jHL6KpQL9p2bb7551mqhXCf3pGHDhoV2We0+ABu1R8rJn8BbRB0QxsE4yY+xDYDFD7to6tWrV9IY9SHlgK8cUKDhq+SV7kRwwHo0ABp4OLRl5wBl19klgqcEb0aUjdASISK24RJ20qYcUA4UxwEFGsXxS+9WDtQoBxRoFM5u8jBIbCW8EWVjxw\/FwijwVRPenyjHrn0pB1zggAINF6SgY1AOKAeUA8oB5UBKOfD\/AZEqWdbP6E9yAAAAAElFTkSuQmCC","height":324,"width":538}}
%---
%[output:2a2be6f3]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAhoAAAFECAYAAABh1Zc8AAAAAXNSR0IArs4c6QAAIABJREFUeF7tfQe4VcXV9hILGIpgN2BBjQWN6IfKVTEqCFGsKBbQgEbsJYoFe4VEDWJDRcQkGkHFrlg+ERuWa0G9Ykk0EoyoKFF+RaKI6P+8k2\/us+\/hlD13Zp+97pl3nodH5MzMnvWuNXvevdaamWV++umnn4SFCBABIkAEiAARIAIZILAMiUYGqLJLIkAEiAARIAJEwCBAokFDIAJEgAgQASJABDJDgEQjM2jZMREgAkSACBABIkCiQRsgAkSACBABIkAEMkOARCMzaNkxESACRIAIEAEiQKJBGyACRIAIEAEiQAQyQ4BEIzNo2XFzEfjxxx+lVatWzW3OdkSACBABIqAIARINRcqIeSiffPKJXHPNNfLss8\/K3LlzZZVVVpGddtpJTjjhBFlvvfVSQ3PLLbfIlClTpE+fPnLMMcfIbbfdJi+99JL0799fdt9999T9lKr4+9\/\/XmbPni1bbLGFGRvKc889J7feeqv5+5lnninrr7+++ftll10mH3zwgfTq1Us22WQT+eMf\/yhrrLGGjB07tmj3r7\/+uowbN05WXXVVGTVqlKnz9ddfC4666dChgyyzzDJy5513muf17t1bBgwYkEqeJ554QiZPntyk7oorrihdu3aVQw45RFZbbbXG34YPHy4fffSRnHTSSbLjjjtW7P+HH36QhQsXmrFhjKXKV199JcOGDTM\/X3\/99fL999\/LxRdfLMsuu6z5f5\/yzTffyJIlS+RnP\/uZLL\/88vLyyy9XxNrnebbtxx9\/HEyGwvF8++23MmHCBHn++eflX\/\/6l2y66abG5g4\/\/PBGnGEvN9xwg\/z85z+XCy+8MIRI7IMIZIIAiUYmsLJTFwSwCIEEgGAUFhCO+++\/Xzp37pyqy4suukhANg4++GABKTjttNPk3nvvFSyglhik6qhEJZCHG2+80RChV155xdS69NJLZfz48ebvl19+uQwcOFCwAG+00UaN\/9a2bVs5\/vjjZc0115QXXnihaO+PPfaYHHfccY11vvzyS9l6661N3VdffVVWXnllOf\/88w15wqJ99tlnpxIFeACXYgVyXHvttVJXV2d+BkH75z\/\/KVdddZXsvffeFft\/\/PHHDaHbYIMNZOrUqSXrz5s3T3r27Gl+nz59uiFQe+yxh\/n\/WbNmVXxOuQp9+\/Y1hA4krV+\/fvLoo49WxNrrgf\/X+N133w0mQ3I88+fPl3322UfmzJmz1DCB85\/\/\/Gfp0qWLWOzxdxB0FiKgFQESDa2aiWhcd911l4wYMcJIjL9369bNLKxDhw41\/3biiSfKKaecshQieCG3b99elltuucbfsEh+\/vnn5isdnoW0RAPhmk8\/\/VRWWmkladeuXUn0QRIOPfRQ8\/uTTz5pvC0gNfiKRsFv+FJ\/6623GhdqLKytW7c2iyH+u+WWWzb2j+divKuvvrpZOJJE49\/\/\/rdsu+22pm4pooExw5MAIlOqWKIBkjNmzBjBM\/E1DoKBxQxt0T\/G1tDQIN99953BLunpwBc2iE8h4UtDNEAygCu+wFEgP8hFkmjAwwHPSKdOnZwtv5BoALdiWKPjBQsWCLxnkKOcnkEUQYZA7kqVQqLhI0PyGeecc47cfvvt5p8uuOAC6d69u0ybNk3+8pe\/GIwOPPBAQ25LEQ1rU8ASOi1WMHdQSuFd6XdnJbFB1AiQaEStfh3CI5SABRAL4VNPPdX4ckSY4L333pNf\/OIXZjGHVwLejSFDhsjbb78tM2bMaCQiICMgHPBiwOVsF\/xConHTTTeZEA0K3M4Ia+A5I0eONC9xlB49esjVV19tXNKFZdGiRcaNjYKv\/j333FM23HBD4+H44osvDElC6AZeB3gf7Jc+whdHHXWUCVdg0UB58MEHBYsKnovF\/te\/\/rXxvgCHSZMmyV577dU4JvQPb8rTTz9t+t51112N5wELKgoWHxCcFVZYYakxW6JR6HVIkiHIi+dBnnfeecfIj\/8HIYB+gDsKxoaF\/dxzzxV4YOBVsbhhjI888ojsv\/\/+hsBANjwb\/w7cLWkq9GgcffTRxkuEgjFCP8AYut9tt93Mv4MAgVQmF3f8vu+++5rxogDDww47zCzMhViDWMF+rBcK9X\/1q1+ZEAsIFbxp22+\/vekHpBYeKsgFW\/jd735n7KSwJMdSSoaDDjrIPHPQoEGN4bA\/\/elPxt6KeYGS5BLjgF3bYvUIOWH7sAV4k6xHA+QI5BF\/bMH4k2GwZ555RvB86AAFIb3Bgwc3kudKvy8FAv+BCKRAgEQjBUiski0CdhG2iwVc9nDl\/8\/\/\/E+TL2iEPrCQFSt4ceOFWS50ghc7whcoICQgL\/BK2NwB\/I4FBwsMFlS8yIst3FjEMGYQHnhdEG7AggevABZYkCDkajz00EOCBQjemkJ3\/syZM417vFjBs+HZwSJuFwTkS5x88sly3333GaJhsWrTpo0hOCijR4+W\/fbbb6kuSxENfIFjoUHBYoT+C0MnWLjhlgdJQH4JZEJBTsDaa68tV1xxReNCD\/KDMaAPOybUxWKHPIxSoRPUsUTN\/h15KB9++OFSRCNJjv7+978bbC0JAsn7zW9+Yzw8yTDV4sWLTT8gZnZx\/dvf\/mb+DlICcvfZZ581Eo3COskwWRLcJNEoJQN0hXwbkIM33njD5KTATuvr6w2BwZ9kAYE44IADzD+BkIKYliqFHg3YivUCQl\/wJEFm+2x4pbbbbjtj37BXkBrgjPLwww8bfZb73RLskgPiD0SgBAIkGjSN3BFAsuN5551nvuILCxYvuI\/h6rZEAy\/fBx54QPDixBcfwhZYMEFCShENEBe83FGSX4o27IGXP8gKXMZYGFHwVYvnF5aJEyea8YKY4CsZixq+8LEIYtG75557zBc1FlssNPhSLiQaZ5xxhtx9991mIUEiKVz0WNTx9WvzOMqFTrB4IFEQRAMhCHg2MI5TTz11qfGWIhqoaMMOWKCBXZJoIBl3q622Mv2dfvrpJvcE4Q\/8waIOj0ex0Mk222xjZIds8FQhvARSU4po2LwWyA4PgMV+nXXWqUg0kPxZKUcDi6j1DODvWDCxwIIoosDuMEbr0YDnCIs97Mnm9YAAFIYZkkSjlAx4lk2qBaGB9wtJnSjwCNk8Hqs01IEXDgWEFUm7aYnGHXfcYcgJngfZkiQatoL+YJco8PqBjEB22DyIGIhJud9ByliIQHMQINFoDmpskwkCyK7HVxkIAbwJyVAGvvAt0UjmbCTzO7DYInxQLBk0OWCbW4F\/++Uvf2meg69Wu2vCfvkWuq5tH9h1gl0fKAjRgEwgpo6vZCzW8JDgRY6CxQhx8kKigR0jCAckyQG8FSAKaYgGvkixiKPYkFKpBNFSRAMED2QJxcpa6NGwrn8rO+ofe+yxhmRgkS9HNOD1sIt5uWTQ5IKKxR5eJXiEdt555yBE48orrzThhGSoAnkMWDihexBZhK0s0YANgtjCDoAHCmwSeTTJkiQapWTAwm3JLLwXsDfoyRLjwomU9O6BQGMHUlqiAX2CvILAvPnmm028SpAJoadkfhD6RUgI4RfIjpyUcr9nMunZaRQIkGhEoWbdQuLrEiEHfAHbr17kQsAtbxdsvMjxVY2vTCxC9ssLL1Z4B\/CFj3BEKY9GEgF4KewuEUs08JVpt6XauvhSLrUlFvH95K4AuMXhyrdf5Ogj+ZxComG\/wpPuc+SKnHXWWamIRpJUWKJxxBFHmLyIwlKKaCQXyuuuu87IWkg0QBBuvvlmk0+S3BVkx12OaEB3lpCVIxo2\/wLjtt4QLP5w49scDXhRkFCa9HoA7zQeDdgR5LP5M3gOFmV80cPzAm8Utj9bomG9FyC+IDtpiEYpGRDKsHoFuYB9Qx9JG07qK5mXksQPdUCQ8W+QGfaLfIpkjobNdUJdyLPxxhsLSBaKJU\/QOfJlsEPIEnn8bj1vlX7X\/Sbh6LQiQKKhVTMRjctuD8WXPNy\/cJnblx8SKlHwAsTXPogGFiO4fHF2Arwc+Aq0i0gpooHcBSxalqBg8dxll12MixwLC77QkQCJL12cOQFXP8Itha5tqxbE3dEHik3wxHkO1i2Of7\/kkkvMORUohUQDu0vw5YmvasiC2D08BUiGLebRwFkgSFostr3VlWgAN4R54KKHFwgkDV\/PcNMniQaIF3Y6IMkWYSXs1kACLMJDNgHREo1kkqslCxZjyF+OaNiwQzJvBW2Bjd3eC88VQlrJxbSQaFiyVIg18kpsLgQ8ZbAvEBckraIgdIWQhg\/RKCUDbCwZjrP2Yxf+wmmOhE54V+BNAaZ\/\/etfTVIyxgt94d8RGgFZKczRsLoD8QYBSYZ+8LzXXnvNkBPYNEgpkmiR4wJPHJJVYe\/lfrdnu0T0aqKogRAg0QgEJLtpPgLJBD\/0AtKAYncT2B0kyWRQLHR4KduvbJvcWekcDevGRnt81eFlbPMa4GVAUiDczih2cS8mWTLxDrkLWGhQrKcCf08m8xUufsn2CNug2ARKSzSSrmwsIlgsQYIKz9FISzRKaSiZi5IkGljYbX4BFn14GLBQ24UJCw8WJhwihQIc4B2wyaBpiQba4mvfJmhCfvSLL3frcUIdjAGeA1ss0UAYCTqD3SD5FsQomQwK7xgWfGsryXwdPBdEBETIh2iUkwG\/2QRi\/B24gjiVKknbQB3gkfQmwbu0+eabL0U0YB+QBWQX+TUgsNamgCe8bpZwwduGUBJsCZ4NkGyUcr+DjLMQgeYgQKLRHNTYJjgCCJ+AJNjtmvYBCBGACCDPwRINxJXh4bAvUSx0CDlggbFEw24nhIsai7M9sCtJatDmyCOPNNtcsc3RFiz8ePHahaeYsEhE3WyzzcxPSc+FfV7hIUqWaCT\/HQs8vDm2IJ8BX9fJOrY\/1EGeAUIHqIOFC7+h2C28lUInSTngxQCJAE7JE0At0bDbW0HE4H5P6gULGQ7HwgIILw621lqSgFwGLPxYGEsRDegah7TZczTwXyRp2kUV7ewOB4QLsPhZNz90jQOrUCzRsKEJ\/BswRHgCRCOJI8aP3B47Trvgw0ODfIzk9lYbpnEJnZSTAc9KeheS9lJqIgFH2GySYEAeEFp7uJrN57BygoRBRhvSQ9gGoUVgByKOsB70imTm5K4g6Au7hXC6a7nfeS1A8NdeNB3WPNHAccp4OWMRQkwWL0cWnQjYg4ZwCBUWQiwAyYOoLNGAaxhfrnAjI1mu3NHXaSUFcUCSJ3ZxwLWOUEY1CrwWWEywtbDUDgMcMoVDtJCjUGy7bTXG+f777xtSgYRIuPKxKNmCfAeEB6A\/7J5pzoIEHPAHOk\/2jWfYw9TQdymMoD+MD4dwldupAaxhX1ickweShcCwnAzJfJhiO1hKPR9jBXHAWGEjlewSusC8wFbkUoe4wROInCfoaa211loq4bTS7yGwYh9xIVDTRMMmjmHC2fMG8PWDGDJLy0MgSTQQg2YhAi0BASSjYjs2CENyt1BLGDvHSARCIFDTRAMuV3xpwTUO1ztczYhVJk8HDAEi+6gOAnALw8Vuk92q81Q+hQj4IWBPW0VIDmG8codw+T2JrYmATgRqlmjYGHrytESbuIb7KhhC0WmQHBURIAJEgAjUFgI1SzQQq0RmNVyV9lRAJLAhmcruGa8tVVIaIkAEiAARIAL6EKhZogGocb8CwiY4XvlnP\/uZubgJGdg4q8Fe8FRMJfYuguRvyOC2pxymVSOSw0IkKqZ9Xh71tMqIjP3kVshi2GzdoYNcdt99qWDTKmeqwaesFIOMgCIGOSljSqNXWg3H3Xfs2FHp6NyHVdNEA+ETkAvs\/ceCj218f\/jDH8qejwAIcVARbq30LfbQHd9+NLfXKqPRYd++ZaFbf+rU1HrWKmdI24hBRuAVg5yUMeTMYF++CNQ00cDBNjiQx57uiAuLcLQxTiAsV0g00puV1hcaiUZ6HdqaWnXpLkn5FjHISRlDWw3780GgpokGtkPCM4GtZdg3jquwcexw4dXMhQCSaKQ3Ka0vNBKN9Dok0XDHSnsLrfMyJG4xyBgSrzz7qmmiAc8FrvO2R0ojORT3IeBoY3o0wpid1slOouGuX626dJeEHo0YdBmDjKFtP6\/+appoWFBxlDCSQctduZxUAD0a6c1R62Qn0UivQ3o03LHS3kLrvAyJWwwyhsQrz76iIBquAJNopEdM62Qn0UivQxINd6y0t9A6L0PiFoOMIfHKsy8SjSLok2ikN0mtk51EI70OSTTcsdLeQuu8DIlbDDKGxCvPvkg0SDS87E\/rZCfRcFerVl26S1K+RQxyUsbQVsP+fBAg0SDR8LEftWcSkGi4qzWGxQmoxCAnZXS3f7bIDgESDRINL+vS+kIj0XBXq1ZduksS1qNR7KTg0GNif7WPQF1dnUyaNKn2BS0iIYkGiYaX4WtdnEg03NWqVZfukoQlGqFytkLLwf5aFgIx2xGJBomG12zVujiRaLirVasu3SUh0QiNGfvzR4BEwx\/DmuohlEHE8OLWKiOJhvuU1KpLd0lINEJjxv78EQi1rviPpPo90KNBj4aX1WldnEg03NWqVZfukmRPNLTnbcScDxDaXkL1R6IRCska6SeUQcTw4tYqI4mG+2TUqkt3SbInGmnsK\/S4XfpzuZnY9vvTTz\/JkiVLZLnlliv7qMWLF1e8xsFlrD51cUP3nDlzZJVVVpGVV17Zp6vM24ZaVzIfaAYPoEeDHg0vs9K6OKVZCFxexlrl9FJeQeMYZITIrnIWWyDS2FdI3bj2VWjb22+\/vblM8qCDDirZ1XPPPSdnnHGGvPDCC0vVueqqq8yCPnz4cOnVq5e8++670rp16yb10P6UU06RV155RT7++GNpaGiQ\/v37uw49Vf2vvvpKzjnnHHnkkUca6++4445y2WWXyZprriljx441F2qOGTMmVX\/JSosWLZL7779f9t57b1lxxRWd25dqQKIRDMra6CiUQbi+0FoielplTLMQkGg0tTitugw9L1zlJNEQ+eijj+T77783C28pooHFH4v7VlttJU888YRcfPHF8uyzz4ZWn8DzcvDBB8uPP\/4oV1xxhayzzjrmuSNGjJAffvhB7rvvPrn22mvlH\/\/4h1x99dXOz58\/f7706NHDEC6QllAl1LoSajzV7IceDXo0vOzN9aXt9TCHxiQaDmD9X1WtunSXpHwLVzlriWjsu+++xqsxaNAgmTBhguDr\/bjjjjMLt\/VIHHDAAeaLHqGIUaNGSffu3eXOO++Uzz\/\/XPbff39DNE488URzJkSHDh3ktNNOk9133914MK6\/\/no599xz5ZBDDjEekN122838G4gHvA1z586V7bbbTi688EL5+c9\/LnfddZfg0svPPvvMhG0++OADGTlypGy++eZGiagHIvHb3\/62Uan19fWCHBn0Cd3Y8t5778nEiRONV+ZPf\/qTIRq\/\/\/3vzVjGjRtnSMM333wjhx56qJEdl2zecMMNcvfddwtCMOjz+OOPN9i8\/PLLsskmm8itt95qiM1FF10kL730kmy66aamf4zvrbfeMrJtueWWBq+kd6WYBZJohJ7JLby\/UAbh+kJribBplZFEw92atOrSXRISjVKhE4QDNttsM+natatZxB9++GGZPHmyCYUg5DFkyBDp27evgGxgsYbX4I477pArr7zSeDVAKkA0tthiC0NQHn\/8cbn33nuN52L27NkmdIK\/\/\/nPf278s8IKKxjCcdhhh8lee+1lwhpff\/21eS7+jr4R9jj66KPlj3\/8o+y0005y8sknG1KA59x2222C0I8tIBMI5WC8pYr1aIC0gCg988wzsvbaa5vnghjAW\/Hpp5\/KUUcdJePHjzek6Oyzz5a\/\/vWvsmDBAhk6dKghEb1795YDDzzQECrUff755+XGG2+UN954w2AGUgIsjzzySEPWypVQ60ro+VCN\/ujRoEfDy860Lk4kGu5q1apLd0lINCoRDbt428V86tSpZuEF0Zg5c6a0bdtWnnrqKTnppJPM\/xcSDXzpg3CAiGAhhxejc+fOjTkaydDJ5ZdfLi+++KIJaaDAawEyM336dENS0Be8Ba1atZJbbrnFEAuMByTm1FNPlddff71JgioIwD333CPTpk3zIhqvvfaanHDCCeZ52267rbz\/\/vsmqXT55ZdvDJ3MmzdP9tlnH0OeunTpYrwbqIuwEDw+IBoYB8hGpUKiUQmhyH4PZRAxvLi1ykii4T5pterSXRISjUpEI7k4Yq7As\/HFF180EgUgCI8BwizIfygkGpaMoB7ICTwOCCfYZNAk0cBivsYaa8h5551nFIMwBbwqDzzwgDz99NNNcik++eQTQ2Awvuuuu07atWtnPC\/J8tBDD5nE1hkzZkinTp0af4K3AiGegQMHGvKA0EmhR+PLL7+Urbfe2ng0VlttNZPXAbIDYoV2CIt89913jUTDkpFCi7rkkktkww03lGHDhhkilqaEWlfSPEtbHXo06NHwskmtixOJhrtaterSXRISjUpEw4YSgFSSaCR3nZQjGsjnQI4FyjbbbGOSL5GIWYxoIDzyzjvvmFAECkIOe+yxh\/kvwhaFSZsI2\/Tr10+uueYaufnmm40HIVlse\/SHkIwtIAwIqYC8gKQkiYYlVsirQPgIRAM5IQjr4A9CIhdccIHxoPz6179uJBrwciCMAo8LSA8KckHgvYFnhkQj3ewk0SDRSGcpJWppXZxINNzVqlWX7pKQaGRNNJCvAFLx4IMPGq+AzfGwROPJJ580\/46Qyd\/+9jcTYkDOB0gJkkJBAm666aaiu0PgjTj\/\/PNNGAOJn8suu+xSCoWX5M033zRkBAmayJnAoo\/nI3HU5miAeNh8Enhe4IlAbgiIBpI3QbjQR\/v27c0YsR13wIABJo\/jscceMwmj8ICcddZZhnCAYOE5+A27U0g00s3OmicaYNkwVsTadt555yautlIQhXJxxfDi1iojiUa6F0CyllZdukuSPdEY3KOH1M+fH3powforPBnUnqNhk0Gb49HA2Rj2HA14GbAzAwWhDSziyXM0sLsEiZ9IokQYBbkeCM8gRNGmTRtDMrCYFzvvAm0xXiSHgqwUK3ifI3kzmacBIgCPDLbgJvvFFle7zRW7brBDBEQDBAb5F3gexoVdJvCwIBwDrwpCM6gHzw5CNbYgIRaJsPh3kBqGTiqbbU0TDcQWYUgoYKwwKGx3qlZ2cAwvbq0ykmhUnvyFNbTq0l2S7IlG6DG1xP5APFZaaaXGkEKhDPjIQ76DDTng\/fuf\/\/zHbFctd\/qoPcciTZIl6oJ0YEdJucO1sJPErgPJcWJ7L7bXduzY0eRsJAvOBYF8KAsXLjQ7U7BF1v6bq85CfcC6PldD\/ZomGsgMButG7A6ZxPj\/KVOmGHdeOUMPZRAxvLi1ykii4f560apLd0lINEJjVq3+Hn30UZPQibwJeBdqqYRaV1oiJjVNNBDHw0EwOBQGBfE6JBchnlcs7mcVGMogYnhxa5WRRMP9daRVl+6SkGiExoz9+SMQal3xH0n1e6hpooEYG2Jt2OcNdxf2QmO\/d\/KUuWKQhzKIGF7cWmUk0XB\/mWjVpbskJBqhMWN\/\/giEWlf8R1L9HmqaaOBoWSQHbbDBBib+hqRQJANVumgneaytVQkSjZDw5FIQ08MhL7VctMrYp08faairKwt99\/r6sof+JBtrlTOkbcUgI\/BylRO2hHyvZOnRY7DMn18fEv6gfcV6Tbzm21yxrpQ7ZCxpAEhIRd5IrZSaJRrYI42sZuyJHj16tNGXPeilUpJRKOYZwxeiVhnp0XB\/RWnVpbsk2Xs0YF99+zYlH6HH6dPf1KnrN5Ije0BWsf6wO8P1htXm3NLK21z\/e15JIWH10XFLaluzRAOZyD179myyywTGjjP1K02uUAYRw4tbq4wkGu6vIa26dJeERKMY0cDFYjhzIllwjDa2drqU5tzSyttcSTR+cjGyllQXe7HtDYQ4yQ2nxeEs\/eQe8mLykGik17LWxYlEI70ObU2tunSXhESjGNHAXSPIVyss2PKPe05w1gTu8sCZEThO\/JhjjjFnXyDUjOO94fnA2RU4gKvULa24nwQJ9\/hyx0fd6aefbk4M5W2uJBo1SzRwkApuA4SrzxaEUfbbb7+ybyISjfSvdq2LE4lGeh2SaJTHqtj7oCWGTkAgkpd\/LbPMMuZQLZCCXXfdVXD5Gc6VuPTSS00uAc6owPXvOJp7vfXWM8cD4Br11VdfvegtrRtttJHssMMO5gp5EJorrrjC3AcCssHbXEk0apZo2NcHJgwOi8GhLvbwmHKvFhKN9IsUiUZ6rLTX1KrL0Li5ylkrRAOJ6Tip0xacJYRTMlFuuOGGxrtIQChwDDeOCselYjjXAgUfbAg\/l7qlFe9ZJNyDnPz73\/82Ieq3337bHC\/A21xJNGqeaLi+qEg00iPm+tJO37NfTXo03PHTqkt3SRg6cQmdAC2bMIqTL3HBGLwdOIcId30U3p6aTAZNEgh4Q0Au\/vKXv5iTNNEXSEkh0Yj1NtdQ60ro+VCN\/mo2GdQHvFAGEcOLW6uMJBruM0CrLt0lIdFwJRq48AxXwYMgIJcNXgnkbMAjYU\/obGhoMLkc8AoXu6UVF43hDhBc\/96tWzeZOHGi2elXSDRivc011LoSej5Uoz8SjSIohzKIGF7cWmUk0XB\/fWjVpbskJBrFiAby0wp3nayxxhom0bN3797m5GRcfT5p0iRBUifuADn00EPNravItfjNb34jhx12mPFUFLulFXkYuBkVx4gjjHL44Yebqx+QhMrbXBk6Yeik4L1EopH+1a51cSLRSK9DW1OrLt0lIdFIe44GwiIIf3z\/\/fdy++23m0vQcPYQduxhNwoOPMRdUSg4uAzeCNxaaolG8pZWeDAOPPBAQzJQQFJuvPFGGTlypCEysd\/mGmpdCT0fqtEfPRr0aHjZmdbFiUTDXa1adekuSfZEY\/DgwSbxUWsJeTIoiAPuhkomkiblTt7Siu2xs2fPNon3SDb4R6bhAAAgAElEQVTFran4L25Wjf02VxINrbMlp3GFMogYXtxaZSTRcJ88WnXpLkn2RCP0mNhfcQRq6TbXUOtKS7QVejTo0fCyW62LE4mGu1q16tJdEhKN0JixP38ESDT8MaypHkIZRAwvbq0ykmi4T0mtunSXhEQjNGbszx+BUOuK\/0iq3wM9GvRoeFmd1sWJRMNdrVp16S5JWKKhPR8jND7sLxsEQubNZDPC7Hol0SDR8LIurYsTiYa7WrXq0l2SsEQj9POr0V8MuoxBxmrYSjWeQaJBouFlZ1onO4mGu1q16tJdEhKNGHQZg4yhbT+v\/kg0SDS8bE\/rZCfRcFerVl26S0KiEYMuY5AxtO3n1R+JBomGl+1pnewkGu5q1apLd0lINGLQZQwyhrb9vPoj0SDR8LI9rZOdRMNdrVp16S4JiUYMuoxBxtC2n1d\/JBokGl62p3Wyk2i4q1WrLt0lIdGIQZcxyBja9vPqj0SDRMPL9rROdhINd7Vq1aW7JCQaMegyBhlD235e\/ZFokGh42Z7WyU6i4a5Wrbp0l4REIwZdxiBjaNvPqz8SDRINL9vTOtlJNNzVqlWX7pKQaMSgyxhkDG37efVXs0QD1x4\/\/\/zzRXFdd911BQtRqRLqqNgYJoJWGUk03F8pWnXpLgmJRgy6jEHG0LafV381SzRwtXGPHj2K4nrSSSfJySefTKIRwOq0TnYSDXflatWluyQkGjHoMgYZQ9t+Xv3VLNEAoEuWLGmC6wMPPCAXXHCBTJ06VdZcc00SjQBWp3Wyk2i4K1erLt0lIdGIQZcxyBja9vPqr6aJRhLUjz\/+WHbbbTe54YYbpFevXmXxZugkvTlqnewkGul1aGtq1aW7JCQaMegyBhlD235e\/UVDNIYMGSJt2rSR8ePHV8SaRKMiRI0VtE52Eo30OiTRcMdKewut8zIkbjHIGBKvPPuKgmi88sorctBBB8m0adOka9euFfEulig6dOhQAVlxKXPmzJEuXbq4NGlxdbXK2KdPH2moqyuLZ\/f6emMTaYpWOdOMPW2dGGQEFjHISRnTWr3Oep06dZKOHTvqHFwzRhUF0TjiiCOkVatWctNNN6WCiB6NVDCZSlq\/KujRSK9DejTcsdLeQuu8DIlbDDKGxCvPvmqeaCA3Y8cdd5Rx48ZJv379UmFNopEKJhKN9DC1iJqxvLhjkJMytogpF80ga55o3HvvvXLaaafJzJkzpW3btqkUS6KRCiYSjfQwtYiaMSxOmr1wIY0kBl3GIGNIm8izr5onGiAZ\/\/jHP+T+++9PjTOJRmqoGDpJD5X6mrG8uGOQkzKqn25RDbDmiUZztEmikR41rS805mik1yFzNNyx0t5C67wMiVsMMobEK8++SDSKoE+ikd4ktU52Eo30OiTRcMdKewut8zIkbjHIGBKvPPsi0SDR8LI\/rZOdRMNdrVp16S5J+RYxyEkZQ1sN+\/NBgESDRMPHfpij4YWersYxLE5APAY5KaOuuRX7aEg0SDS85oDWFxo9Gu5q1apLd0no0YhBlzHIGNr28+qPRINEw8v2tE52Eg13tWrVpbskJBox6DIGGUPbfl79kWiQaHjZntbJTqLhrlatunSXhEQjBl3GIGNo28+rPxINEg0v29M62Uk03NWqVZfukpBoxKDLGGQMbft59UeiQaLhZXtaJzuJhrtaterSXRISjRh0GYOMoW0\/r\/5INEg0vGxP62Qn0XBXq1ZduktCohGDLmOQMbTt59UfiQaJhpftaZ3sJBruatWqS3dJSDRi0GUMMoa2\/bz6I9Eg0fCyPa2TnUTDXa1adekuCYlGDLqMQcbQtp9XfyQaJBpetqd1spNouKtVqy7dJSHRiEGXMcgY2vbz6o9Eg0TDy\/a0TnYSDXe1atWluyQkGjHoMgYZQ9t+Xv2RaJBoeNme1slOouGuVq26dJeERCMGXcYgY2jbz6s\/Eg0SDS\/b0zrZSTTc1apVl+6SkGjEoMsYZAxt+3n1R6JBouFle1onO4mGu1q16tJdEhKNGHQZg4yhbT+v\/kg0SDS8bE\/rZCfRcFerVl26S0KiEYMuY5AxtO3n1R+JBomGl+1pnewkGu5q1apLd0lINGLQZQwyhrb9vPqreaKxZMkSeeONN2TGjBnSs2dP6d69e0WszSI1a1bFepUqxDARtMpIolHJOpf+Xasu3SUh0YhBlzHIGNr28+qvponGjz\/+KIMHD5a3335bNtxwQ2loaJCBAwfK5ZdfXhZvEo305qh1spNopNehralVl+6SkGjEoMsYZAxt+3n1V9NEY+LEiXLppZfKY489Jp07d5bHH39cjjnmGHnppZdktdVWK4k5iUZ6c9Q62Uk00uuQRMMdK+0ttM7LkLjFIGNIvPLsq6aJxtChQ6Vt27Zy3XXXyYIFC6Rdu3by+eefS6dOnaR169YkGgEsT+tkJ9FwV65WXbpLQo9GDLqMQcbQtp9XfzVNNLbffntZb731ZObMmbJw4ULZdtttZdiwYbLrrrsydBLI4rROdhINdwVr1aW7JCQaMegyBhlD235e\/dUs0fj2229ls802M7hedtllss4668iECRNk2rRp8sQTTwgWolKl2G\/wjgwZMsRJT3PmzJEuXbo4tWlplbXK2KdPH2moqysLZ\/f6emMPaYpWOdOMPW2dGGQEFjHISRnTWr3OevC6d+zYUefgmjGqmiUaP\/zwg2y00Uay7777ypgxYww08+bNMztPkAyKpNByRIO7TtJZk9avCno00ukvWUurLt0loUcjBl3GIGNo28+rv5olGgAUoZNddtlFRo0a1YRoXHXVVbL33nuTaASwOq2TnUTDXbladekuCYlGDLqMQcbQtp9XfzVNNK699loZP368XHnllbL55pvL2LFjZdKkSfLCCy\/ImmuuSaIRwOq0TnYSDXflatWluyQkGjHoMgYZQ9t+Xv3VNNFAAuiIESPkkUceMfhiB8oVV1wh\/fr1K4s3t7emN0etk51EI70ObU2tunSXhEQjBl3GIGNo28+rv5omGhZUbGmdP3++2YFSblurrU+ikd4ctU52Eo30OiTRcMdKewut8zIkbjHIGBKvPPuKgmi4AkyikR4xrZOdRCO9Dkk03LHS3kLrvAyJWwwyhsQrz75INIqgT6KR3iS1TnYSjfQ6JNFwx0p7C63zMiRuMcgYEq88+yLRINHwsj+tk51Ew12tWnXpLkn5FjHISRlDWw3780GARINEw8d+ROsLjUTDXa1adekuCYlGDLqMQcbQtp9XfyQaJBpetqd1spNouKtVqy7dJSHRiEGXMcgY2vbz6o9Eg0TDy\/a0TnYSDXe1atWluyQkGjHoMgYZQ9t+Xv2RaJBoeNme1slOouGuVq26dJeERCMGXcYgY2jbz6s\/Eg0SDS\/b0zrZSTTc1apVl+6SkGjEoMsYZAxt+3n1R6JBouFle1onO4mGu1q16tJdEhKNGHQZg4yhbT+v\/kg0SDS8bE\/rZCfRcFerVl26S0KiEYMuY5AxtO3n1R+JBomGl+1pnewkGu5q1apLd0lINGLQZQwyhrb9vPoj0SDR8LI9rZOdRMNdrVp16S4JiUYMuoxBxtC2n1d\/JBokGl62p3Wyk2i4q1WrLt0lIdGIQZcxyBja9vPqj0SDRMPL9rROdhINd7Vq1aW7JCQaMegyBhlD235e\/ZFokGh42Z7WyU6i4a5Wrbp0l4REIwZdxiBjaNvPqz8SDRINL9vTOtlJNNzVqlWX7pKQaMSgyxhkDG37efVHokGi4WV7Wic7iYa7WrXq0l0SEo0YdBmDjKFtP6\/+SDRINLxsT+tkJ9FwV6tWXbpLQqIRgy5jkDG07efVX4shGi+88IKsuuqqstFGGzXBavHixfLQQw\/J7rvvLiuuuGKT37799lupr69v8m+oU1dXVxZvs0jNmuWtkxgmglYZSTTczVerLt0lIdGIQZcxyBja9vPqTz3RmDNnjnz\/\/fcyfPhw2W677eSAAw5ogtXf\/\/53Of744+XRRx+VjTfeuMlvb731luy9995N\/q1Lly7y7LPPkmgEsjitk51Ew13BWnXpLgmJRgy6jEHG0LafV3\/qicb2228vc+fOLYtP27Zt5fnnn5cOHTo0qQfyMXr0aJk2bZoTvvRopIdL62Qn0UivQ1tTqy7dJSHRiEGXMcgY2vbz6k890XjllVdk0aJFcumll8rmm28ue+65ZxOsVlhhBenWrZu0a9duKQzHjRsn8Gqg7f\/7f\/9P1lprLVl22WUrYk2iURGixgpaJzuJRnodkmi4Y6W9hdZ5GRK3GGQMiVeefaknGhacGTNmyMorryxdu3ZNjdeZZ54pkydPbqwPz8fVV18tvXv3ZugkNYot8+uQRMNdwbG8uGOQkzK62z9bZIdAiyEaP\/30kwmBvPvuu\/LFF18shcjpp58uIBLJ8rvf\/U6++eYbOeecc2ThwoUmjDJ9+nTzp3PnziVRxSJVWIYOHSpDhgxx0gTyS5ATUstFk4zI42loaKgI99YdOsjN3bpJ9\/r61GE1TXJWFLCZFWKQEdDEICdlbOYkUNKsU6dO0rFjRyWj8R9GiyEaY8eOlTFjxhiJsXgvv\/zyTaS\/\/\/77pX379mURscmh8GrstddeZYkGd52kMy5NX04giH37\/ne30O233y6DSpC8qd8NlVl9+8r6U6em3l2kSc50mnGvFYOMQCUGOSmju\/2zRXYItBii0b9\/f5Or8cgjj0jr1q0rIoKtrePHj5c+ffqY3A4U7FDBNtgJEyaUDZ8wR6MivI0VNL3QSDTS661YTU269JOkfOsY5KSMWVoQ+3ZFoEURjQEDBsiRRx6ZWsajjz7ahFrg7UC45fLLLzfnajz33HOy0kor0aORGsnSFTW90Eg0\/BSqSZd+kpBoxKDLGGTMch5Us+8WQzQQOrnvvvvkgQceKLrDpBho8GCMGDFC3nzzTfPzKqusYpJBsWW2XKFHI70JaprsJBrp9UaPRvqkcj9U82mtaV5mhUAMMmaFXbX7bTFEY+LEiXLeeecZstCzZ09p06ZNI1atWrWSiy66aKmTQW2FTz\/91IRdkNux3HLLVcSYRKMiRI0VNE12Eo30eiPRINHws5b8W2t69+SPhu4RtBiiASLx+uuvl0Tz1ltvXerAruZCT6KRHjlNk51EI73eSDRINPysJf\/Wmt49+aOhewQthmhUE0YSjfRoa5rsJBrp9UaiQaLhZy35t9b07skfDd0jaDFEAxenYW94qfLb3\/421W6UNOog0UiD0n\/raJrsJBrp9UaiQaLhZy35t9b07skfDd0jaDFE46ijjpIXX3yxEU0cwGUL8jaefPLJiudopFUFiUZapEg00iOlv2YsL+4Y5KSM+udbTCNsMUSjUClI7vzwww9l5MiRJgn0xhtvDKY3Eo30UGp6odGjkV5v9GjQo+FnLfm31vTuyR8N3SNosUTDwgpjw6FcL730kqy22mpB0CbRSA+jpslOopFebyQaJBp+1pJ\/a03vnvzR0D2CFk80HnvsMTnuuOPknnvuka222ioI2iQa6WHUNNlJNNLrjUSDRMPPWvJvrendkz8aukfQYojGTTfdZI4QT5bvvvtOnnnmGfNPzz\/\/PLe35mBrmiY7iYafAWjSpZ8k5VvHICdlzNKC2LcrAi2GaOAG1sJzNHBQ1yabbCK4WfWXv\/ylq+wl69OjkR5KTS80Eo30eqNHgx4NP2vJv7Wmd0\/+aOgeQYshGtWEkUQjPdqaJjuJRnq9kWiQaPhZS\/6tNb178kdD9whaFNGYP3++3HHHHfL+++\/LkiVLpGvXrrLPPvuY\/4YsJBrp0dQ02Uk00uuNRCPsO8MP+fCtNc3L8NL9t8cYZMwKu2r322KIxuzZs2WvvfYSnJ+BczNQcCMrCm5lHThwYDDsSDTSQ6lpspNopNcbiQaJhp+15N9a07snfzR0j6DFEI0TTjjBXPl+8803y3rrrWdQnTt3rlx\/\/fVy2223yYwZM6RTp05B0CbRSA+jpslOopFebyQaJBp+1pJ\/a03vnvzR0D2CFkM0ttlmGzn55JPlkEMOaYLot99+K5tttplMmjRJ6urqgqBNopEeRk2TnUQjvd5INEg0\/Kwl\/9aa3j35o6F7BC2GaPzqV78yu0uOOOKIJoh+\/fXXsuWWW8qECROkd+\/eQdAm0UgPo6bJTqKRXm8kGiQaftaSf2tN75780dA9ghZDNC655BKZPHmy4Lr4nj17mlNAZ86cKePHj5epU6cydJKTnWma7CQafkagSZd+kpRvHYOclDFLC2Lfrgi0GKKBEAm8GfX19UvJePXVV5tE0VCFHo30SGp6oZFopNcbPRr0aPhZS\/6tNb178kdD9whaDNGwMIJovPPOO4Ktrl26dBGEVNZaa62gKJNopIdT02Qn0UivNxINEg0\/a8m\/taZ3T\/5o6B5BiyAab731lnz00Uey++67N6IJL0aPHj1khx12kGWWWaYiyl999ZVJGB08eLCstNJKZeuTaFSEs7GCpslOopFebyQaJBp+1pJ\/a03vnvzR0D0C9UTjyiuvlGuvvdbc0Ir7TmzBkeM4UwMejbFjx0q7du3KIo3tsY888og88cQTggWpXCHRSG+0miY7iUZ6vZFokGj4WUv+rTW9e\/JHQ\/cIVBONJ598UoYNGya77LKLnHvuuU1OAJ03b57cddddMnr0aBk0aJCMGjWqJNJ33323nHHGGeZ3Eo2wBqlpspNo+OlWky79JCnfOgY5KWOWFsS+XRFQTTTghcDtrLhMbbnllisq27hx48zJoC+\/\/LKsuuqqS9WZNWuW7LrrrjJy5EhDVkg0XE2k5by0STT8dBvD4gSEYpCTMvrNBbYOi4BqotG\/f3\/p1q2b8VqUKpZIFDuwa9GiRbL\/\/vsbj8iAAQMM4SDRCGtAml5oJBp+utWkSz9JWg45zkrOGHQZg4xZ2Ue1+1VNNA466CDp3LmzjBkzpiQuOJZ8jz32kFtvvVV69erVpN6ll14qzz33nNx7770yZ84cJ6JR+EAcFjZkyJCK+hk+fLg0NDRUrNe9e\/eyclXsQEkF4IrdPxoK8njq6v6L\/ZQpU2TgGmsUHdbTS4ZJQ12ddK+vl2nTpqUauiY5Uw24GZVikBGwxCAnZWzGBFDUBNdpdOzYUdGI\/Iaimmgg7wK3tWIxWH311YtKapNFQSh+\/vOfN9b54IMPpG\/fviZ\/Y+ONNxbkdFx33XVyzDHHSL9+\/cxpoqWKTzKoadu3r+l6wYIF0r59+6KPWX\/qVIE3pqUXTV8V9Gj4WZMmXfpJUr51DHJSxiwtiH27IqCaaNiwCMInF1xwgeC+E1u++eYb48VAWAWHdWG7a7KgLTwayfo4g2PbbbeVAw88UPbbbz8SDVdrKVJf0wuNRMNPoZp06ScJiUYMuoxBxiznQTX7Vk00AARyKk455RSzlbVt27ay7rrrCk4JhZGhwGsBQlHp5tZ\/\/etfsvPOO2eeo0GPRjXNt+mzSDT8sI\/lxR2DnJTRby6wdVgE1BMNiPvll1+aw7ZwIugnn3wiq6yyimywwQbGO4EEzzQFB37ttNNOJBppwHKoo+mFRqLhoDjl3ik\/SejR0DQvs9JlDDJmhV21+20RRKPaoDBHIz3imiY7iUZ6vRWrqUmXfpKQaMSgyxhkzHIeVLNvEo0iaJNopDdBTZOdRCO93kg0eDKon7Xk31rTuyd\/NHSPgESDRMPLQjVNdhINL1VGcZAVENJks34aK92aMmaFLPttDgIkGiQazbGbxjaaXmgkGl6qjGIBJtHwsxFNrTW9ezThonEsJBokGl52qWmyk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/JBokGl42p2myk2h4qZJEww8+Va01zcusgIlBxqywq3a\/NU80Zs+eLa+\/\/rost9xy0qtXL+nUqVNFjLFgzZo1q2K9YhVM2759zU8LFiyQ9u3bF+1n\/alTm\/2MZg0so0aaJjuJhp+SNenST5LyrWOQkzJmaUHs2xWBmiYazz33nAwZMsRgsuaaa8rcuXNl9OjRst9++5XFiUQjvRlpeqGRaKTXW7GamnTpJwmJRgy6jEHGLOdBNfuuaaIxePBgadeunYwaNUpWXXVVOeuss2TGjBny+OOPyzLLLFMSZxKN9CaoabKTaKTXG4lGVz+wlLfWNC+zgioGGbPCrtr91izR+Omnn+SYY46Rww8\/XOrq6gyul1xyiTz44INSX18vyy67LIlGAGvTNNlJNPwUqkmXfpLQoxGDLmOQMct5UM2+a5ZoJEGEB+PVV1+VCRMmyJlnnilHHXUUQyeBrEzTZCfR8FOqJl36SUKiEYMuY5Axy3lQzb6jIBrjx4+XO++802TVIz\/jsssuq+jRKFTC0KFDG\/M9yimoT58+0vB\/HpSFCxdK27Zti1bvXl8v06ZNq6auM3nWnDlzpEuXLpn07dopsK+razDNpkyZIgPXWKNoF08vGWZ05KIDTXK64pK2fgwyAosY5KSMaa1eZz1sWujYsaPOwTVjVDVLNBA6WbJkidltYstdd90lI0aMkHvuuUe22mqrknAxRyO9JWn6qqBHI73eitXUpEs\/SejRiEGXMciY5TyoZt81SzQ+\/vhj2XHHHQXkokePHgZTGCa+eseOHSv9+\/cn0QhgaZomO4mGn0I16dJPEhKNGHQZg4xZzoNq9l2zRAMgDhgwQFZccUU5\/fTTzfkZ11xzjdx\/\/\/0yffp06dy5M4lGAEvTNNlJNPwUqkmXfpKQaMSgyxhkzHIeVLPvmiYaOEfj2GOPFeRKoCBfAvkZ5bwZqMfQSXoT1DTZSTTS642hE25v9bOW\/Ftrevfkj4buEdQ00QD0ixYtkg8\/\/NBoYd1115XWrVtX1AiJRkWIGitomuwkGun1RqJBouFnLfm31vTuyR8N3SOoeaLRHPhJNNKjpmmyk2ik1xuJBomGn7Xk31rTuyd\/NHSPgESjiH5INNIbrabJTqKRXm8kGiQaftaSf2tN75780dA9AhINEg0vC9U02Uk0vFTJ21v94FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq90uiQaLhZXOaJjuJhpcqSTT84FPVWtO8zAqYGGTMCrtq91vzRGPevHny8ssvy2effSZdu3aVnXbaSVq1alUWZyxYs2bNapYuTNu+fU3bBQsWSPv27Yv2s\/7Uqc1+RrMGllEjTZOdRMNPyZp06SdJ+dYxyEkZs7Qg9u2KQE0Tjffee0\/2339\/adOmjXTo0MF8se2xxx5y7bXXkmi4WkqJ+ppeaCQafkrVpEs\/SUg0YtBlDDJmOQ+q2XdNE41zzjlHGhoa5K677pIVV1xR7rjjDjn77LNl+vTp0rlz55I406OR3gQ1TXYSjfR6K1ZTky79JCHRiEGXMciY5TyoZt81TTT69+8ve+21lxx77LEG09mzZ0vv3r3luuuuk913351EI4ClaZrsJBp+CtWkSz9JSDRi0GUMMmY5D6rZd00TjSSQP\/74o4waNUomT54s9fX10rZtWxKNAJamabKTaPgpVJMu\/SQh0YhBlzHImOU8qGbfURANGOSFF15oQiaXX365DBw4sCzGWLAKy9ChQ2XIkCEVddOnTx9pqKsz9RYuXFiS0HSvr5dp06ZV7E97hTlz5kiXLl1UDBPY19U1mLFMmTJFBq6xRtFxPb1kmNGRiw40yZkV2C4yjhgwQF79+utmD6V79+4yZsyYZrf3aegip89zsm47fPhwExquVPLEutLYfH6vFT0Ww6BTp07SsWNHH3hUta15onHffffJqaeeKr169TJkoxiJKNQIczTS26imr18KGrgAABxISURBVAp6NNLrrVhNF10md1cl+8Juqr5tbjH\/dPucOTJo0KCig5o6tfk7u\/yklJrZxltKB8AnueOtVna4FerdxV59bYbt\/RCoaaIxadIkOffcc2X06NGy3377pUaKRCM1VKpe2iQa6fVGotHVDywFrUk0\/mmOLGDRj0BNE42+ffvKWmutJSeeeGITTWywwQay8sorl9QOiUZ6w9X0VUGikV5vJBotf4Ei0SDR8Jvx1Wtds0Tj66+\/li233LIokpU8HCQa6Q2QRCM9VtpruuiSoZP8tUmiQaKRvxWmG0HNEo104hevRaKRHj2XxSl9r82rSY9G83CzrVx0SaLhh3WI1iQaJBoh7KgafZBoFEGZRCO96bksTul7bV5NEo3m4Uai4YdbXq1JNEg08rI91+eSaJBouNpMk\/okGl7wqWrsokt6NPJXHYkGiUb+VphuBCQaJBrpLKVELZfFyetBKRrTo5ECpDJVXHRJouGHdYjWJBokGiHsqBp9kGiQaHjZmcvi5PWgFI1JNFKARKKhaku2j8ZINEg0fOynmm1JNEg0vOyNRMMLPlWNXXRJj0b+qiPRINHI3wrTjYBEg0QjnaUwdFLzhwORaHhNhao3JtEg0ai60TXzgSQaJBrNNJ3\/NnNZnLwelKIxQycpQGLoRJXN+miMRINEw8d+qtmWRINEw8veSDS84FPV2EWXDJ3krzoSDRKN\/K0w3QhINEg00lkKQycMnSRsgETDa9oEaUyiQaIRxJCq0AmJBomGl5m5fAV7PShFY4ZOUoDE0AlDJ35moqa1pnePGlCUDoREg0TDyzQ1TXYSDS9VOi3A9Gj4YR2iNT0a9GiEsKNq9EGiQaLhZWckGl7wqWrsoksSjfxVR6JBopG\/FaYbAYkGiUY6S2GOBnM0mKPhNVdCNybRINEIbVNZ9UeiQaLhZVsuX8FeD0rRmKGTFCAxR8MpROSHaLatSTRINLK1sHC9k2iQaHhZE4mGF3yqGrvokqGT\/FVHokGikb8VphsBiQaJRjpLYeiEoROGTrzmSujGJBokGqFtKqv+SDRINLxsy+Ur2OtBKRozdJICJIZOGDrxMxM1rTW9e9SAonQgJBokGl6mqWmyk2h4qdJpAWboxA\/rEK3p0aBHI4QdVaOPaIjGvffeKz179pTOnTtXxNVM4FmzKtYrViE5+RcsWCDt27cv2s\/6U6c2+xnNGlhGjUg0MgI2h25ddEmikYOCCh5JokGikb8VphtBFERj+vTpMnToULn11lulV69eFZEh0agIUWMFl8Upfa\/Nq0mPRvNws61cdEmi4Yd1iNYkGiQaIeyoGn3UNNF499135aSTTpIPPvjAYEmiEd6kXBan8E9v2iOJhh\/CLrok0fDDOkRrEg0SjRB2VI0+appozJ07V1588UWZP3++jBw5kkQjA4tyWZwyeHyTLkk0\/BB20SWJhh\/WIVqTaJBohLCjavRR00TDAjhv3jyTn0GPRniTclmcwj+dHo2QmLrokkQjJPLN64tEg0SjeZZT\/VYkGkUwxwQuLMjxGDJkSEUN9enTRxrq6ky9hQsXStu2bYu26V5fL9OmTavYn\/YKc+bMkS5duqgYJrCvq2swY5kyZYoMXGONouN6eskwoyMXHWiSMyuwXWRM2nlyPMB052UnmH+6+7PPZM899yw63Pr67rnZv4ucWWEdot9SOkDfh8+cKa8tXFjxMd27d5cxY8ZUrKexQp56HD58uDQ0\/PddU640F99OnTpJx44dK3XfYn4n0ShBNLjrJJ0Nu3wFp+ux+bUYOmk+dmjpokt6NPywDtG6nEcDu9r6trnFPOb2OXNk0KBBRR85dWrzd9iFkMGnDxd79XlOsbbJd025vlsyviExI9Eg0fCypzwne+HASTS8VEmi4Qdf1VuTaOQXOiHRcDN3Eg0SDTeLKahNouEFn6rGLrqkRyN\/1ZFokGjkb4XpRkCiQaKRzlJK1HJZnLwelKIxPRopQCpTxUWXJBp+WIdoTaJBohHCjqrRRxREwxVIHtiVHjGXxSl9r82rSaLRPNxsKxddkmj4YR2iNYkGiUYIO6pGHyQa9Gh42ZnL4uT1oBSNSTRSgESPhlMuih+i2bYm0SDRyNbCwvVOokGi4WVNJBpe8Klq7KJLejTyVx2JBolG\/laYbgQkGiQa6SylRC2XxcnrQSka06ORAiR6NOjRSNhAS95+mee7h7tO3N41JBokGm4WU1A7z8leOHASDS9VOi3A9Gj4YR2iNT0a9GiEsKNq9EGiQaLhZWckGl7wqWrsoksSjfxVR6JBopG\/FaYbAYkGiUY6S2HoRLp27eqFlfbGJBraNdR0fCQaJBotxWJJNEg0vGzVZXHyelCKxgydpACJORpOISI\/RLNtTaJBopGthYXrnUSDRMPLmkg0vOBT1dhFlwyd5K86Eg0SjfytMN0ISDRINNJZCkMnDJ0kbIBEw2vaBGlMokGiEcSQqtAJiQaJhpeZuXwFez0oRWOGTlKAxNAJQycJG+D21ubNGW5vdcONRINEw81iCmqTaHjBp6qxiy7p0chfdfRo0KORvxWmGwGJBolGOkth6IShE4ZOvOZK6MYkGiQaoW0qq\/5INEg0vGzL5SvY60EpGjN0kgIkhk4YOmHoxG+iiAhDJ24QkmiQaLhZDEMnXnhpbuxCGhk6yV+T9GjQo5G\/FaYbAYkGiUY6S2HohKEThk685kroxiQaJBqhbSqr\/kg0SDS8bMvlK9jrQSkaM3SSAiSGThg6YejEb6IwdOKMH4kGiYaz0SQbkGh4waeqsYsuGTrJX3X0aNCjkb8VphsBiQaJRjpLYeiEoROGTrzmSujGJBokGqFtKqv+ap5oLFq0SOrr6+WLL76QbbbZRtZee+2KWJoJPGtWxXrFKiQn\/4IFC6R9+\/ZF+1l\/6tRmP6NZA8uokctXcEZDaOyWoRM\/hF10SY+GH9YhWpNokGiEsKNq9FHTROPrr7+W3XbbTbDgr7766iY2O2HCBOndu3dZbEMRjRkzZkiPHj1qmmhcdNFFcsEFF1TDVis+I0uioUnOikA0s4KLjC2ZaLjI2Uwoq9IsdqKRpx65vdXNxGuaaJx\/\/vny9NNPy8MPP2w8C5dccolMnjxZQABWWGGFkkiFIhq33367DBo0qKaJhg9WbqZauXaWREOTnJWRaF4NFxlbMtFwkbN5SFanVexEI089kmi42XhNE41f\/epXZqE\/9thjDSovv\/yyHHzwwXLnnXeaMEqp4mPAyclPouFmjL61STT8EHSxexINP6xDtCbRaH6I2xd\/Eg03BGuWaCA3Y9NNN5Vx48ZJv379DCqffvqp7LDDDjJ27Fjp378\/iYabrRT3zHjkswR4fJMuSDT8ECXR8MOv2q1JNEg0qm1zzX1ezRKNjz\/+WHbccUe59dZbpVevXgafxYsXy8YbbyxXX3217LXXXiUxGzx4sEkgbW7punhxxab\/XH75inVYwR2BxYu7NjZq\/+OPRTv4rvWH7h2zRWoE2ixa19Rd0KpVyTbLL\/\/P1P2xYvMQoB6ah1vaVsl3Tak2zbXz3\/3ud4I\/tVJqlmggEXTLLbeUyy+\/XAYOHGj0NX\/+fJOc+eCDD8rmm29eKzqkHESACBABIkAE1CJQs0QDiP\/yl7+U3\/zmN3LGGWcYBcBLAW\/F22+\/LSuuuKJapXBgRIAIEAEiQARqBYGaJhqXXXaZ3HjjjfLAAw\/ImmuuKSeccIIsu+yyMmnSpFrRH+UgAkSACBABIqAagZomGt9++60cc8wxMn36dKOErl27ysSJEw3pYCECRIAIEAEiQASyR6CmiYaFD7tNvvvuu5o\/Qjp7c+ETiAARIAJEgAi4IRAF0XCDhLWJABEgAkSACBCBUAiQaIRCkv0QASJABIgAESACSyFAokGjIAJEgAgQASJABDJDgEQjM2jZMREgAkSACBABIkCiQRsgAkSACBABIkAEMkOARMMT2h9\/\/FFmzpwpf\/vb32SzzTYzf5ZZZpmyvb7wwgvmwLCtttrK8+nZNZ89e7a89NJLsu6665pxtm7duuzDFixYIA899JAceOCBstxyyzXWnTdvnrnM7rPPPjO7fnbaaSdpVeZo6uwkWrrnUDLanr\/66itzRgsOhVtppZWqKUrZZ4WSc8mSJfLGG2+Y24979uwp3bt3rzkZf\/jhB3OwH+x25513lk6dOqmRMTkQl\/cOtvljDn7wwQdmaz8um2zXrp1KuTAoF3uFTK+\/\/rrZVQh7xCGNtmh+96gFP6OBkWh4AnvUUUfJE088Id26dZN33nnHnNthTyIt1jWMv3fv3vLb3\/5WTjnlFM+nZ9P8T3\/6k4wcOVI22WQTQ6Dq6urk5ptvLnua6hVXXCHXXXddk1NX33vvPdl\/\/\/2lTZs20qFDB\/nnP\/8pe+yxh1x77bXZDNyh11AyJh+JA+EeeeQRYw+48EpDCSUnFjZ7qu6GG24oDQ0N5mh\/HPGfdwkl46xZs2SfffYx4rRv317mzp0rv\/\/9782Nz9pK2vcOrmIYOnSo0Zd9R4FsTJs2TeXpyC66xIcN7gPBBwzk\/OKLL8x7C3aq+d2jzZaqMR4SDQ+UsahgcXn44YfNTbGPP\/64IRrF7lLBbbJDhgyRV155xTzxxBNPVEk0PvroI+N1uOSSS+SQQw4RezndBRdcYF5YheXee++V0aNHm5cySvJ493POOce84O666y7zUrvjjjvk7LPPNgeode7c2QN5v6YhZbQjufvuuxsJphaiEVJOHHR36aWXymOPPWZ0Z20dXq\/VVlvNTyEerUPKePHFF5svf9j08ssvL\/j\/KVOmyIsvvtjES+cx3CBNXd47djF++umnZZ111jEL8G677SZjxoyRfffdN8h4QnXiqkt4ZvDRduGFFwo8UfjAe\/XVVwWynnfeeSrfPaGwamn9kGh4aAwLKSYHbohFsVfTn3baaXLcccc16RluZxAQFHwlgXVr9GjYF1OSMOCLboUVVmiUMynYW2+9Je+\/\/75xpyNskGzXv39\/c0vusccea5rAJYoXAzwfu+++uwfyfk1DyoiR4Et41113NV9T5557rhqPRkg5QTLbtm1rdIcwGVzvn3\/+uQktVAqr+WmrfOuQMuKjASE+EGOUq666ynjyEC7C1QVaist7B++av\/\/973LLLbc0Dn+bbbYRzM2LLrpIi0hmHC66\/PLLL2XrrbeWO++8UyAPCggi3r0IfR122GEq3z2qAK\/iYEg0PMA+6KCDzC2wYM+2JFl2qa779Okje+65p0qicfXVV5uvuKlTpzYOH4snvurgbi1VHn30UTn++ONLXlgH1\/uoUaNk8uTJ5kWARSuvElJGkEuEh3bZZRcZMGCAIRxaPBoh5dx+++1lvfXWM\/lICxculG233VaGDRtm5M2zhJQRZPmAAw4wsX7k2Dz77LOGOCLMqak0970DGew8BZnCTdaaSnN1CRm++eYbc4EmQrRJUoXfNL17NOFdzbGQaHigjZcvFplTTz21sRd8\/W+wwQZmUW2JROPMM880X0D33Xdf4\/CRU4EXE168zSEayM2AexMhE8T0EdvPs4SUEeGE5557znxNzZkzRxXRCCUnkgmR5IyCiwrhgp8wYYIhnnmTqlAyQjYb\/sL8RTgIhBjhBYQZNJXmvHeQw4CF\/M9\/\/rPJOYGnQ1tpri4x\/xCShZ3CLpNJytrePdowr9Z4SDQ8kIZXAgmTyFGwBS5JfHEUy2ewdTR7NLCQ4IVrc0kwZpCmf\/3rX+YmXFeiAcICItarVy9DNjQkSYaSERnvffv2lUGDBsnGG29sdiogtIA8nX79+smWW27pYV3+TUPJifj3Rhtt1GTRhazYeZI3cQwlI0Kb0Nevf\/3rxvlsXfkgVEg41FJc3zsIbx5++OEmTwq5V8jB0lhcdQm7RLgSoWvkvyEUndztpfHdoxH3aoyJRMMD5eHDh5v4\/P333296gftuiy22kNtuu03w1VGqaCYaiHmeddZZJi4NNyQKiBPkQYa3C9FAzgZczyBi++23nwfSYZuGkhG6h0fDFugfX8EIK2Cbb94yh5IT8kH\/CA9ZT50lGshj2HvvvcMqyKG3UDJaeZK7TGwi9NixY01Og5bi8t5BqAs7aZCzMGLEiFzzaSrh56pLvI9wVEChFwPP0fruqYRBrf5OouGh2SeffNLEqfFVhy9YhBiwswKuPDBrEBAkLRXGeDUTjX\/\/+99moTz00ENNYhVkQZLc7bffbr5gsWcdOw5OOumkJtvjiuVo4Gt\/rbXWMjtskgWu6ZVXXtkDeb+mIWVMjgReH5y9kHc4wY4ppJyw7fHjx8uVV15p8pKw+OJljhc9tkvmVULKCDIFuwSZws4aeKcQ73\/mmWdk7bXXzkvEpZ7r8t6BBwN5UTfddFOThFaEhpBzo6m46BJhEiSDHn300SbBPFngmQIx1Pju0YR3NcdCouGJtj0\/At0gwRHhBevNwAKLGCGSK5MFRAO7MU4++WTPp2fT3L7IbO\/YNoZwQPJLAYQj6aYsJBqICZcKHWjwcISQsRB9uz1PC9HA+ELJiQRQfBFja6W1ddg+CHbeJZSM+Pr\/4x\/\/aMi1LRpstRi+ad87CLPgfJ\/CAo9b0huXtw7t89Pq8rXXXpMjjjii6LBx3ADO6ylWtOpTC\/5ZjYNEIwCy2O6HRMBf\/OIXqvbb+4j2\/fffm5ME8dWD2G4tlhhkhN5CyoktrfPnzzd2kee21kJ7DCkj5MO5MPBiaD5BsxbfO6HttRbfWy1RJhKNlqg1jpkIEAEiQASIQAtBgESjhSiKwyQCRIAIEAEi0BIRINFoiVrjmGsGAWw9vP766wX\/RSgCB2DtuOOOst1221W8nM+CgBMecZkUdrlgqx\/yKexprD5AIeETp7kib8FeFGiPesaOJGxXtgU7brBbCac04jh+jOMPf\/hD0dADEqRxZgKSjbFttrDgNyQku+7aQZIyjkgPfe4FtrkiJwDH8LMQASLgjgCJhjtmbEEEgiDw7rvvmqS1VVZZxSRVYicAEkmRvHf66aenJgvY3YPzH7BwY+sjFvK\/\/OUv3mO0CYfJ5FYkAeOQOpTknTXYfYKdSiAfq666qkkcxQFvxW4\/tdtGsWsFF\/ahDe6osEnTSKbGlkz0kbaA6Njtt6ETVJHYvMMOO5idGxgvCxEgAm4IkGi44cXaRCAYArizAtuGcZEXFmcUeAxwABhuzX3zzTeX8mr89NNPZT0dIYkGdl\/gIKRkpj68HNZjkDysC9tfsfUVu49weFm5Ukg0gAMwsMfeJ4lGJXntc+AVwtZyXKjVqlUrLx3hyOrCPiAbdtzAa2K9O14PYWMiEBECJBoRKZui6kIAV33jDhl77oodHQ79wtkN2P784Ycfmi97HBuNkw6xIONQOLjxt9pqK9MEF\/jhTAGc11JINBDCwAKMrda4yA5bcHG2Ap4LbwNu8sTJrcV2kCAEg5BM8shq3OeCMyZw+RjOzwDBQMFJuAgvgBxBHpARnEGBEAuIBW5ChdcDJ+nC44CwCjwaGAeuBsez8BuOuoeHBncGYScJyEebNm0M4Sk8jyWpTYwTONiQEbY+AqdPP\/3U3K6Mq+2BDbxI8PZgxwZkOf\/8881Osa+++spsTYdXBTvIELrBeRo48wXF3npqvTC6LImjIQK6ESDR0K0fjq6GEcABWDjLoFu3buYyL1xyhcUWC58tyVAFjljGIn\/DDTeYrcf2inZ4AEAicLlfkmjA1Y8F3YZhsJjiKHgQDCzEOHYbRzgjLwRjKVZAMnBqJo7hxrZPjBFHRWP7J9pgfCjJI8pxUB3GgXAIzpZBSAP1QaywwMOLg4JFG7eiIkQDeTB+yIEjslEfYQqElnDrMY7ET97UmRwr6gIDnBBpD2\/C\/+PfIS\/+gFzg\/xGmAmnB3zEOECU8wx5\/DeKBMeJOEFwVb88NWbx4sfHU4IAol5BODZsvRSMCqREg0UgNFSsSgbAIYKHHQjdx4kT54osvGjtHEiS+znEvjCUa+H\/kYKDgEDgc+obkyyOPPNIssoVEA4s0QhsIS9iDjXCpFv787\/\/+rznzBQXH5WNxRT5Fly5dlhIQIQmETkBqQBxwQy9yM3CeBnI17rnnHrMg4xhyG2JJEg2EMjBukBJ70yuOLb\/mmmsM0cA4i4VOOnbsaK4NBxEBQYL3BrkcIAmFxYZ4kAhqk0uBCfqAhwJhEHhNQKpAvHCkPm7dRdIqvCQgcPCgwKuD+vgvPB\/ACR4S6+0B5jg51F4jH9Ya2BsRqF0ESDRqV7eUrIUggDyE999\/35AKuPmxcOLLGwsdQidY0BGGwG4UW7C7AyEIuPcLiYa9ewd1k0eEg5TAM5FcrD\/55BOTgIoQC778C4slOiAKOLURhAN9IJcEiz++8OEhgTfCkpUk0cAx3vAONDQ0SPv27U338E5gsS9HNOCZADGwBaQLd1sUu2\/Hkoi333678XA5YAJig5ANCm7XBeFJJrACQxAkjN3mmMADg8RaeFXwx973gz4gKwhIuVuMW4jJcZhEoKoIkGhUFW4+jAj8FwHkJCCxEgsqFrxksSGPcePGmZ0oIBrIz7DXXxfeNFqMaOAuGngNkIMBrwQKFncs8rifp7AgdGO9HMnfkiED3OqLsWD7KQoWfdzvgptN4e2wC3CSaCA0hHYgUvBOoGArLxb4ckSjcNdJOaJhd8cUEo1kH5ZozJgxo3EnTJJoWAKEnBAQKXiNEMayHhErL+Sw4RTaMhEgAukQINFIhxNrEYGgCGBnAxIUcfkTCEeyIBl08ODB5lIvJFxiccdiiuTF5EKNm3GRAFpINBCGgQcEuRw4AwP9ILQCgoALtpLeBXyhP\/DAA6af1VdfvaiM1hOCH5OeD7t449\/hJbHnaiSJBsgEEkPhqUGoAsV6IEIRDYwJzy4MnaQlGsi5wK4S6AJJpSggE7hMMClvsevZgxoFOyMCNYoAiUaNKpZi6UcAB22BEMAljy98uOmxuwEhA5zdgMUOOyBANBBKQfgCBAW5F\/BM4Msb3oRSyaDIQ8Diib4QHkHf8Grsu+++JrcDO1BARFDvqaeeanK7ZxI9jBFjRUl6DRB2seEWS2ZQJ0k08Gx4bRDGQC4IvBkIVYAMWaJhbxgFCQIZKXaORjmPhr0KPZkHUthHJY8Gds385z\/\/EVwgiKTPm2++2YRTQPosAcMYbF6MfuviCImAHgRINPTogiOJDAFcdY3dDvhqThYkHSIsgevYbY4Ekiax6KEgaRPeCZAIlEKigeRJLJQo2A6LnSM4mwP5HDb50z4PRAWhlXJnX9iDxbB7xPZr22MrKnaMIHSCREkU7BLB1lz7b0jqTOZWgFhhK6klGvawL7R94403TLjHJXRi80WQqGpvGS5FNJK3DidDJyBu2OYLYmcLcjqQhItid7YU5spEZrIUlwg0CwESjWbBxkZEIBwCIAY46htls802K7q9FYv3uuuuK\/AiYGeFz6FROEUT5AC38iIvw6evtCjYZ66zzjpFTwvF78g9WWmlldJ22aQeCBvOHvHJn4C3COeAYBwYJ\/JjbAHBwh\/sollhhRWaNUY2IgKxIkCiEavmKXeLQMB6NEA04OFgKY4Ajl3HLhF4SuDNCFkQWkKICNtwEXZiIQJEwA0BEg03vFibCFQVARKN9HAjDwOJrQhvhCzY8YPDwnDAVzW8PyHHzr6IgAYESDQ0aIFjIAJEgAgQASJQowj8f5pgOxJp\/v4SAAAAAElFTkSuQmCC","height":324,"width":538}}
%---
%[output:3fe4c203]
%   data: {"dataType":"text","outputData":{"text":"Generated 2 visualization figures.\nGenerating comprehensive summary...\nEnhanced E\/I classification complete!\n\n--- Final Classification Summary ---\nTotal Neurons Analyzed: 26\n----------------------------------\nExcitatory (Putative):   9 (34.6%)\nInhibitory (Putative):   17 (65.4%)\nE\/I Ratio:               0.53\n----------------------------------\n","truncated":false}}
%---
%[output:345a75e1]
%   data: {"dataType":"text","outputData":{"text":"\n=== Layer-Aware E\/I Analysis ===\nAssigning cortical layers to all valid neurons...\nAnalyzing spike width distribution for each layer:\n  - Layer L5: Found 26 valid neurons.\n","truncated":false}}
%---
%[output:2f9fe308]
%   data: {"dataType":"warning","outputData":{"text":"Warning: Hartigan's Dip Test not found. Assuming unimodal distribution."}}
%---
%[output:2b69fb5d]
%   data: {"dataType":"text","outputData":{"text":"\n--- Layer-Aware Bimodality Summary ---\n    <strong>Layer<\/strong>    <strong>NeuronCount<\/strong>    <strong>Dip_pValue<\/strong>    <strong>GMM_SilhouetteScore<\/strong>    <strong>GMM_Threshold_ms<\/strong>    <strong>IsBimodal<\/strong>\n    <strong>_____<\/strong>    <strong>___________<\/strong>    <strong>__________<\/strong>    <strong>___________________<\/strong>    <strong>________________<\/strong>    <strong>_________<\/strong>\n\n    \"L5\"         26             1               0.80197                NaN             false  \n\nGenerating layer-stratified plots...\n","truncated":false}}
%---
%[output:3969cd2f]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAhoAAAFnCAYAAADg7eCkAAAAAXNSR0IArs4c6QAAIABJREFUeF7t3Q9QpVX9x\/GPsqG0oqMlURsVWBIBRUOoLX9MRCIK05WSP6FZTUGh9BcpjfqFm0KbSbIJK04TE4xgbUJhZW5DsilhlLqR\/RNKY9jaiAJ3gFD3N+fU3WAF7gXvMR+e95nZGd099yzn9XwP+7nnOc\/luCNHjhwRDQEEEEAAAQQQcCBwHEHDgSpDIoAAAggggIAVIGhQCAgggAACCCDgTICg4YyWgRFAAAEEEECAoEENIIAAAggggIAzAYKGM1oGRgABBBBAAAGCBjWAAAIIIIAAAs4ECBrOaBkYAQQQQAABBAga1AACCCCAAAIIOBMgaDijZWAEEEAAAQQQIGhQAwgggAACCCDgTICg4YyWgRFAAAEEEECAoEENIIAAAggggIAzAYKGM1oGRgABBBBAAAGCBjWAAAIIIIAAAs4ECBrOaBkYAQQQQAABBAga1AACCCCAAAIIOBMgaDijZWAEEEAAAQQQIGhQAwgggAACCCDgTICg4YyWgRFAAAEEEECAoEENIIAAAggggIAzAYKGM1oGRgABBBBAAAHfBI29e\/fq7LPP1rZt27jqCCCAAAIIIPAcCfgiaAwODuryyy9XR0eHsrKyniNa\/hoEEEAAAQQQ2NRB45FHHtFVV12lRx991F5pggYFjwACCCCAwHMrsKmDxsGDB3X\/\/fdrenpa1113HUHjua0t\/jYEEEAAAQS0qYNG4PoeOnTIns8IdUejrKxMQ0NDy8qjpqZG5hcNAQQQQAABBEIXIGisYJWQkKCxsbHQFemJAAIIIIAAAisKEDQIGiwNBBBAAAEEnAkQNAgazoqLgRFAAAEEECBoEDRYBQgggAACCDgTIGgQNJwVFwMjgAACCCDgi6Cx3svMYdD1itEfAQQQQACBlQUIGuxosDYQQAABBBBwJkDQIGg4Ky4GRgABBBBAgKBB0GAVIIAAAggg4EyAoEHQcFZcDIwAAggggABBg6DBKkAAAQQQQMCZAEGDoOGsuBgYAQQQQAABggZBg1WAAAIIIICAMwGCBkHDWXExMAIIIIAAAgQNggarAAEEEEAAAWcCBA2ChrPiYmAEEEAAAQQIGgQNVgECCCCAAALOBAgaBA1nxcXACCCAAAIIEDQIGqwCBBBAAAEEnAkQNAgazoqLgRFAAAEEECBoEDRYBQgggAACCDgTIGgQNJwVFwMjgAACCCBA0CBosAoQQAABBBBwJkDQIGg4Ky4GRgABBBBAwJNBY2FhQUNDQ5qamlJGRobi4uJWvZKPP\/64HnjgAW3ZskVnnXWWYmNjg171hIQEjY2NBe1HBwQQQAABBBBYW8BzQWNmZkYFBQWanZ1VTEyMxsfH1d7ertzc3GfM1ASMSy+9VFu3btWJJ55og0l3d7cNJ2s1ggbLBgEEEEAAgfAIeC5o1NfXa2BgQP39\/YqOjlZDQ4N6eno0MjKiyMjIZSo1NTWanp7WLbfcohNOOEF1dXX6yU9+Ync4CBrhKSBGQQABBBBAYC0BzwWNnJwclZaWqqqqys5reHhYJSUlz9ipmJubU3Jysnbt2qUdO3bYviZkXHHFFbrvvvvWvIXCjgaLBgEEEEAAgfAIeCpomLMZSUlJam1tVX5+vhWYnJxUZmamWlpaVFhYeFTlyJEjysvL00UXXaQrr7zS\/r55XVNTk775zW9q+\/btqwqaoHFsM7sj5hcNAQQQQAABBEIX8FTQmJiYUHZ2tjo6OpSVlWVnubi4qMTERDU3N6uoqGjZzL\/2ta\/Z2yYVFRV64QtfqLa2Nh0+fFi33367PRi6WmNHI\/QCoicCCCCAAAJrCXgqaJiDoGlpaXZXori42M7LnMFIT09XX1+fUlJSls3V3D4x4cKc6Tj55JNtSLn++uv1s5\/9TKeffjpBg7WBAAIIIICAYwFPBQ1jkZqaancoamtrLY15zLWsrEyjo6OKiopaxmXCx2tf+1qdeeaZ9vf37t2rz33uczpw4MCarOxoOK46hkcAAQQQ8I2A54JGY2Oj3aXo7e21Bzqrq6sVERGhrq4ue9HMLRRzCNSczzB\/Zj4Pw\/Q1QeSqq67SJZdcEvSsBUHDN\/XPRBFAAAEEHAt4LmiY2yGVlZUaHBy0NPHx8ers7Dz6FInZ8SgvL7ePspqdi89+9rN6+OGHbV8TPnbv3q0XvOAF7Gg4LiyGRwABBBBAwAh4LmgELpt52mR+ft4GjWDtscces4dBX\/ziFwfrav+cHY2QmOiEAAIIIIBAUAHPBo2gM3sWHQgazwKPlyKAAAIIILBEgKCxQjkQNFgjCCCAAAIIhEeAoEHQCE8lMQoCCCCAAAIrCBA0CBosDAQQQAABBJwJEDQIGs6Ki4ERQAABBBAgaBA0WAUIIIAAAgg4EyBoEDScFRcDI4AAAgggQNAgaLAKEEAAAQQQcCZA0CBoOCsuBkYAAQQQQICgQdBgFSCAAAIIIOBMgKBB0HBWXAyMAAIIIIAAQYOgwSpAAAEEEEDAmQBBg6DhrLgYGAEEEEAAAYIGQYNVgAACCCCAgDMBggZBw1lxMTACCCCAAAIEDYIGqwABBBBAAAFnAgQNgoaz4mJgBBBAAAEECBoEDVYBAggggAACzgQ8GTQWFhY0NDSkqakpZWRkKC4ublWgJ5980vY9dOiQ3vKWt+jUU08NipmQkKCxsbGg\/eiAAAIIIIAAAmsLeC5ozMzMqKCgQLOzs4qJidH4+Lja29uVm5v7jJmasPDOd77T\/n50dLQOHjyoL37xiyopKVlThaDBskEAAQQQQCA8Ap4LGvX19RoYGFB\/f78NDw0NDerp6dHIyIgiIyOXqXzhC1\/Q8PCw9u7dqxe84AUy\/\/+9731P999\/v7Zs2bKqIEEjPMXFKAgggAACCHguaOTk5Ki0tFRVVVX26pkgYXYouru77W2Upa26ulp\/+ctfdMcdd9jfvummm3TbbbfpwQcfVEREBEGD+kcAAQQQQMCxgKeChjmbkZSUpNbWVuXn51uayclJZWZmqqWlRYWFhcu4zC7Hu971Lr3hDW\/QKaeconvvvVfXXnut3ve+963Jyo6G46pjeAQQQAAB3wh4KmhMTEwoOztbHR0dysrKshdpcXFRiYmJam5uVlFR0bIL961vfUu1tbU644wzdPrpp9tDoRdddJFuvPHGoEHj2A41NTUyv2gIIIAAAgggELqAp4KGOQialpampqYmFRcX21lOT08rPT1dfX19SklJOTrzp556yvZ961vfql27dtnf\/+53v2vDwr59+xQfH7+qEjsaoRcQPRFAAAEEEFhLwFNBw0wkNTVVFRUVdqfCNLNLUVZWptHRUUVFRR2dq3mc9eyzz172lElgR2Sl2yxLkQgaLBoEEEAAAQTCI+C5oNHY2Ki2tjb19vYqNjZW5sCnOdjZ1dVlRcwtlOTkZOXl5Wn79u067bTTtHPnTm3btk27d+\/WN77xDf3kJz9Z87M3CBrhKS5GQQABBBBAwHNBY25uTpWVlRocHLRXz9wC6ezstKEjsONRXl6uuro6HThwQF\/60pe0f\/\/+o1fa3EbZsWPHmleeoMHCQAABBBBAIDwCngsagWmbp03m5+fXPGsR6GvOcZgP6zKfIHrSSScFlSNoBCWiAwIIIIAAAiEJeDZohDS7DXYiaGwQjpchgAACCCBwjABBY4WSIGiwThBAAAEEEAiPAEGDoBGeSmIUBBBAAAEEVhAgaBA0WBgIIIAAAgg4EyBoEDScFRcDI4AAAgggQNAgaLAKEEAAAQQQcCZA0CBoOCsuBkYAAQQQQICgQdBgFSCAAAIIIOBMgKBB0HBWXAyMAAIIIIAAQYOgwSpAAAEEEEDAmQBBg6DhrLgYGAEEEEAAAYIGQYNVgAACCCCAgDMBggZBw1lxMTACCCCAAAIEDYIGqwABBBBAAAFnAgQNgoaz4mJgBBBAAAEECBoEDVYBAggggAACzgQIGgQNZ8XFwAgggAACCBA0CBqsAgQQQAABBJwJEDQIGs6Ki4ERQAABBBDwZNBYWFjQ0NCQpqamlJGRobi4uGdcyX\/961\/66U9\/uuIVfuUrX6mEhIRVr775s7GxMaoDAQQQQAABBJ6lgOeCxszMjAoKCjQ7O6uYmBiNj4+rvb1dubm5yyimp6eVnp6+Is9VV12lj370owSNZ1k8vBwBBBBAAIFgAp4LGvX19RoYGFB\/f7+io6PV0NCgnp4ejYyMKDIyctl8n3rqqWX\/39vbq8997nP60Y9+pNjYWIJGsOrgzxFAAAEEEHiWAp4LGjk5OSotLVVVVZWd+vDwsEpKStTd3W1vo6zWJiYm7E7ILbfcoqysrDXZuHXyLKuKlyOAAAIIIPAfAU8FDXM2IykpSa2trcrPz7dTmJycVGZmplpaWlRYWLjqhb3ssst04oknas+ePUEv\/krnN2pqamR+0RBAAAEEEEAgdAFPBQ2zK5Gdna2Ojo6juxKLi4tKTExUc3OzioqKVpz5Aw88oEsvvVT79u1TfHx8UB12NIIS0QEBBBBAAIGQBDwVNMxB0LS0NDU1Nam4uNhOMHDos6+vTykpKStO+v3vf7+OP\/543XrrrSGhEDRCYqITAggggAACQQU8FTTMbFJTU1VRUaHa2lo7OfOYa1lZmUZHRxUVFfWMCQd2QZbebgmmQtAIJsSfI4AAAgggEJqA54JGY2Oj2traZJ4gMU+OVFdXKyIiQl1dXXbG5hZKcnKy8vLy7P\/v3btXn\/zkJ3XgwAFt3bo1JBWCRkhMdEIAAQQQQCCogOeCxtzcnCorKzU4OGgnZ85cdHZ2Hn1c1ex4lJeXq66uzv65CRl\/+MMfdOeddwbFCHQgaIRMRUcEEEAAAQTWFPBc0AjMxjxtMj8\/H9LhzvXWAEFjvWL0RwABBBBAYGUBzwYNlxeUoOFSl7ERQAABBPwkQNBY4WoTNPy0BJgrAggggIBLAYIGQcNlfTE2AggggIDPBQgaBA2fLwGmjwACCCDgUoCgQdBwWV+MjQACCCDgcwGCBkHD50uA6SOAAAIIuBQgaBA0XNYXYyOAAAII+FyAoEHQ8PkSYPoIIIAAAi4FCBoEDZf1xdgIIIAAAj4XIGgQNHy+BJg+AggggIBLAYIGQcNlfTE2AggggIDPBQgaBA2fLwGmjwACCCDgUoCgQdBwWV+MjQACCCDgcwGCBkHD50uA6SOAAAIIuBQgaBA0XNYXYyOAAAII+FyAoEHQ8PkSYPoIIIAAAi4FCBoEDZf1xdgIIIAAAj4XIGgQNHy+BJg+AggggIBLAU8GjYWFBQ0NDWlqakoZGRmKi4tb1eipp57Sgw8+qJGREZ199tl6wxveENQzISFBY2NjQfvRAQEEEEAAAQTWFvBc0JiZmVFBQYFmZ2cVExOj8fFxtbe3Kzc39xkzffrpp1VWVqbR0VG9+tWv1kMPPaTi4mI1NTWtqULQYNkggAACCCAQHgHPBY36+noNDAyov79f0dHRamhoUE9Pj92xiIyMXKbS2dmpG264QT\/4wQ+0bds23X333aqsrNTPfvYznX766asKEjTCU1yMggACCCCAgOeCRk5OjkpLS1VVVWWv3vDwsEpKStTd3W1voyxtl19+ubZu3ardu3fbHZCTTjpJf\/3rX3XqqafqhBNOIGhQ\/wgggAACCDgW8FTQMGczkpKS1Nraqvz8fEszOTmpzMxMtbS0qLCwcBnX9u3b9apXvUoHDhzQ4cOHddZZZ+kDH\/iA8vLygt46ObZDTU2NzC8aAggggAACCIQu4KmgMTExoezsbHV0dCgrK8vOcnFxUYmJiWpublZRUdHRmc\/NzSk5Odn+f2Njo17xilfYsxz79u3TPffcI3N7ZLXGrZPQC4ieCCCAAAIIrCXgqaBhDoKmpaXZw5zmUKdp09PTSk9PV19fn1JSUo7O9cknn9SZZ56piy66SDfeeKP9\/UOHDtknT5a+fiUcggaLBgEEEEAAgfAIeCpomCmnpqaqoqJCtbW1VsA85hp4siQqKmqZirl1ct5552nnzp3LgsZNN92kCy+8kB2N8NQQoyCAAAIIILCqgOeChrkN0tbWpt7eXsXGxqq6uloRERHq6uqykzS3UMwtE3MO4+abb9aePXv0la98xe52mHMcpt99991nX7taY0eDFYMAAggggEB4BDwXNMzZC\/OI6uDgoBWIj4+XeYw1EBzMjkd5ebnq6ursAdCrr75ad911l+1rnkD58pe\/fPQgKUEjPEXEKAgggAACCKwm4LmgEZiIedpkfn7eBo1gzTzSas5ymCdQ1nqsNTAOOxrBRPlzBBBAAAEEQhPwbNAIbXob60XQ2Jgbr0IAAQQQQOBYAYLGCjVB0GChIIAAAgggEB4BggZBIzyVxCgIIIAAAgisIEDQIGiwMBBAAAEEEHAmQNAgaDgrLgZGAAEEEECAoEHQYBUggAACCCDgTICgQdBwVlwMjAACCCCAAEGDoMEqQAABBBBAwJkAQYOg4ay4GBgBBBBAAAGCBkGDVYAAAggggIAzAYIGQcNZcTEwAggggAACBA2CBqsAAQQQQAABZwIEDYKGs+JiYAQQQAABBAgaBA1WAQIIIIAAAs4ECBoEDWfFxcAIIIAAAggQNAgarAIEEEAAAQScCRA0CBrOiouBEUAAAQQQIGgQNFgFCCCAAAIIOBPwZNBYWFjQ0NCQpqamlJGRobi4uBWB5ubmbL+lLSoqSuecc86aoAkJCRobG3OGzsAIIIAAAgj4RcBzQWNmZkYFBQWanZ1VTEyMxsfH1d7ertzc3Gdcs1\/96le68MILl\/3+y1\/+ct17770EDb9UOPNEAAEEEPifCnguaNTX12tgYED9\/f2Kjo5WQ0ODenp6NDIyosjIyGWY3\/\/+97Vr1y7t27dvXcjsaKyLi84IIIAAAgisKuC5oJGTk6PS0lJVVVXZSQ0PD6ukpETd3d32NsrS1traKrOrccMNN+gf\/\/iHXvrSlyoiIiJoORA0ghLRAQEEEEAAgZAEPBU0zNmMpKQkmQCRn59vJzg5OanMzEy1tLSosLBw2aTr6ursbkegbd26Vc3NzSveZln6QoJGSLVDJwQQQAABBIIKeCpoTExMKDs7Wx0dHcrKyrKTW1xcVGJiog0QRUVFyyZcU1OjJ554Qtdcc40OHz5sb6MMDg7aX9u2bVsVxwSNY5sZy\/yiIYAAAggggEDoAp4KGuYgaFpampqamlRcXGxnOT09rfT0dPX19SklJWXNmQcOh64UStjRCL1o6IkAAggggECoAp4KGmZSqampqqioUG1trZ2jeXy1rKxMo6OjMo+uBpp5tHXPnj06\/\/zzjwaQ3\/72t3rb29626lMqgddy6yTU8qEfAggggAACawt4Lmg0Njaqra1Nvb29io2NVXV1tT3g2dXVZWdqdiuSk5OVl5enD33oQ3rkkUd055132s\/cMDshJpjs379fp5xyyqoyBA2WDQIIIIAAAuER8FzQMDsVlZWV9pyFafHx8ers7LShI7DjUV5eLnMQ1OxgXH311Xr44Yftn73oRS+yQWT79u1r6hE0wlNcjIIAAggggIDngkbgkpmnTebn523QCNZMX\/PEivmwri1btgTrLoJGUCI6IIAAAgggEJKAZ4NGSLPbYCeCxgbheBkCCCCAAALHCBA0VigJggbrBAEEEEAAgfAIEDQIGuGpJEZBAAEEEEBgBQGCBkGDhYEAAggggIAzAYIGQcNZcTEwAggggAACBA2CBqsAAQQQQAABZwIEDYKGs+JiYAQQQAABBAgaBA1WAQIIIIAAAs4ECBoEDWfFxcAIIIAAAggQNAgarAIEEEAAAQScCRA0CBrOiouBEUAAAQQQIGgQNFgFCCCAAAIIOBMgaBA0nBUXAyOAAAIIIEDQIGiwChBAAAEEEHAmQNAgaDgrLgZGAAEEEECAoEHQYBUggAACCCDgTICgQdBwVlwMjAACCCCAAEGDoMEqQAABBBBAwJmAJ4PGwsKChoaGNDU1pYyMDMXFxQUF+uc\/\/6muri6VlZXplFNOWbN\/QkKCxsbGgo5JBwQQQAABBBBYW8BzQWNmZkYFBQWanZ1VTEyMxsfH1d7ertzc3DVnWl1drbvuukv33HOPTJBYqxE0WDYIIIAAAgiER8BzQaO+vl4DAwPq7+9XdHS0Ghoa1NPTo5GREUVGRq6o8q1vfUu1tbX2zwga4SkcRkEAAQQQQCAUAc8FjZycHJWWlqqqqsrOb3h4WCUlJeru7ra3UY5t5hZIXl6errvuOl177bUEjVCqgj4IIIAAAgiEScBTQcOczUhKSlJra6vy8\/MtweTkpDIzM9XS0qLCwsJlLKb\/JZdcovPOO08XX3yxDRzsaISpchgGAQQQQACBEAQ8FTQmJiaUnZ2tjo4OZWVl2ektLi4qMTFRzc3NKioqWjblG264Qfv379fevXv15z\/\/eV1B41i7mpoamV80BBBAAAEEEAhdwFNBwxwETUtLU1NTk4qLi+0sp6enlZ6err6+PqWkpByd+aOPPqoLLrjA3mYxQeTQoUPavXu3Kisr7W6IGWe1xmHQ0AuInggggAACCKwl4KmgYSaSmpqqioqKo4c7zWOu5pHV0dFRRUVFHZ2rOZthdjQC7YknnrCPxJ511ll697vfrR07dhA0WBsIIIAAAgg4FvBc0GhsbFRbW5t6e3sVGxsr89hqRESE\/YwM08wtlOTkZHubZGl77LHH9Ja3vIUzGo4LiuERQAABBBBYKuC5oDE3N2dvfwwODtp5xMfHq7Oz04aOwI5HeXm56urqll3pxx9\/XOeeey5Bg\/pHAAEEEEDgORTwXNAI2JinTebn523QCHfjjEa4RRkPAQQQQMCvAp4NGi4vGEHDpS5jI4AAAgj4SYCgscLVJmj4aQkwVwQQQAABlwIEDYKGy\/pibAQQQAABnwsQNAgaPl8CTB8BBBBAwKUAQYOg4bK+GBsBBBBAwOcCBA2Chs+XANNHAAEEEHApQNAgaLisL8ZGAAEEEPC5AEGDoOHzJcD0EUAAAQRcChA0CBou64uxEUAAAQR8LkDQIGj4fAkwfQQQQAABlwIEDYKGy\/pibAQQQAABnwsQNAgaPl8CTB8BBBBAwKUAQYOg4bK+GBsBBBBAwOcCBA2Chs+XANNHAAEEEHApQNAgaLisL8ZGAAEEEPC5AEGDoOHzJcD0EUAAAQRcChA0CBou64uxEUAAAQR8LkDQIGj4fAkwfQQQQAABlwIEDYKGy\/pibAQQQAABnwt4MmgsLCxoaGhIU1NTysjIUFxc3KqX8Y9\/\/KN++ctfasuWLcrKytKpp54a9JInJCRobGwsaD86IIAAAggggMDaAp4LGjMzMyooKNDs7KxiYmI0Pj6u9vZ25ebmPmOm+\/fv12WXXWZ\/PzY2VgcPHtSuXbu0Y8eONVUIGiwbBBBAAAEEwiPguaBRX1+vgYEB9ff3Kzo6Wg0NDerp6dHIyIgiIyOXqZSVlemkk07Szp079eIXv1if\/vSnbb+7775bxx133KqCBI3wFBejIIAAAggg4LmgkZOTo9LSUlVVVdmrNzw8rJKSEnV3d9vbKIF25MgRVVZW6oorrtA555xjf9uEkr6+PnvbJSIigqBB\/SOAAAIIIOBYwFNBw5zNSEpKUmtrq\/Lz8y3N5OSkMjMz1dLSosLCwhW5zA7Gz3\/+c3uLpa6uTh\/84AfXZDU7Gse2mpoamV80BBBAAAEEEAhdwFNBY2JiQtnZ2ero6LAHO01bXFxUYmKimpubVVRUtOLM9+zZY3c8zHkOcz6jsbGRHY3Qa4SeCCCAAAIIbFjAU0HDHARNS0tTU1OTiouL7aSnp6eVnp5ub4mkpKQchTC3Tp566in7tEmg3XHHHbr66qv17W9\/W2984xtXReOMxobriRcigAACCCCwTMBTQcN85ampqaqoqFBtba2diDlvYQ59jo6OKioq6ujkArsfJlyYIGKa2dE4\/\/zz17zNYvoRNFglCCCAAAIIhEfAc0HD3PZoa2tTb2+vfWS1urra3gbp6uqyIuYWSnJysvLy8nTxxRfb8PGpT33Kfn7GV7\/6Vd15550aHBzUtm3b2NEITw0xCgIIIIAAAqsKeC5ozM3N2adJTFgwLT4+Xp2dnTZ0BHY8ysvL7aFP8zka5umUw4cP2z\/bunWrPZ+x2qHRgBI7GqwYBBBAAAEEwiPguaARmLZ52mR+ft4GjbWaeVLlT3\/6k+3yyle+UieccEJQOYJGUCI6IIAAAgggEJKAZ4NGSLPbYCeCxgbheBkCCCCAAALHCBA0VigJggbrBAEEEEAAgfAIEDQIGuGpJEZBAAEEEEBgBQGCBkGDhYEAAggggIAzAYIGQcNZcTEwAggggAACBA2CBqsAAQQQQAABZwIEDYKGs+JiYAQQQAABBAgaBA1WAQIIIIAAAs4ECBoEDWfFxcAIIIAAAggQNAgarAIEEEAAAQScCRA0CBrOiouBEUAAAQQQIGgQNFgFCCCAAAIIOBMgaBA0nBUXAyOAAAIIIEDQIGiwChBAAAEEEHAmQNAgaDgrLgZGAAEEEECAoEHQYBUggAACCCDgTICgQdBwVlwMjAACCCCAAEGDoMEqQAABBBBAwJmAJ4PGwsKChoaGNDU1pYyMDMXFxa0KdOjQIQ0PD+svf\/mL4uPjde655+r4449fEzQhIUFjY2PO0BkYAQQQQAABvwh4LmjMzMyooKBAs7OziomJ0fj4uNrb25Wbm\/uMa\/a73\/1Ol1xyiU488USdfPLJtu\/b3\/523XzzzQQNv1Q480QAAQQQ+J8KeC5o1NfXa2BgQP39\/YqOjlZDQ4N6eno0MjKiyMjIZZjXXHONHnroId1xxx2KiorS7bffrs985jMaHBzUtm3bVoVnR+N\/WpP85QgggAACm0jAc0EjJydHpaWlqqqqspfB3BYpKSlRd3e3vY2ytBUWFqqoqOho3z\/+8Y9252P37t1629veRtDYRIXMVBBAAAEEnp8Cngoa5mxGUlKSWltblZ8wI0XFAAAP8ElEQVSfb0UnJyeVmZmplpYWmWCxWnv66ae1c+dOu\/thznds3bqVoPH8rEm+KgQQQACBTSTgqaAxMTGh7OxsdXR0KCsry16GxcVFJSYmqrm52e5erNTM2YzPf\/7z9pZJU1OTiouL17yE5tbJsa2mpkbmFw0BBBBAAAEEQhfwVNAwB0HT0tKWhYXp6Wmlp6err69PKSkpz5j5d77zHX3iE5+wwcSEjZVCxLEv4oxG6AVETwQQQAABBNYS8FTQMBNJTU1VRUWFamtr7bzMbZCysjKNjo7aA59LW1dXl6699lrt2rVLO3bsCLkSCBohU9ERAQQQQACBNQU8FzQaGxvV1tam3t5excbGqrq6WhERETKhwjRzCyU5OVl5eXm64IIL9NKXvlRXXnnlMoQzzjhDp5122qowBA1WDQIIIIAAAuER8FzQmJubU2VlpT1vYZr5EK7Ozk4bOgI7HuXl5frwhz9sb7Os1ILtcBA0wlNcjIIAAggggIDngkbgkpmnTebn523QCHcjaIRblPEQQAABBPwq4Nmg4fKCETRc6jI2AggggICfBAgaK1xtgoaflgBzRQABBBBwKUDQIGi4rC\/GRgABBBDwuQBBg6Dh8yXA9BFAAAEEXAoQNAgaLuuLsRFAAAEEfC5A0CBo+HwJMH0EEEAAAZcCBA2Chsv6YmwEEEAAAZ8LEDQIGj5fAkwfAQQQQMClAEGDoOGyvhgbAQQQQMDnAgQNgobPlwDTRwABBBBwKUDQIGi4rC\/GRgABBBDwuQBBg6Dh8yXA9BFAAAEEXAoQNAgaLuuLsRFAAAEEfC5A0CBo+HwJMH0EEEAAAZcCBA2Chsv6YmwEEEAAAZ8LEDQIGj5fAkwfAQQQQMClAEGDoOGyvhgbAQQQQMDnAgQNgobPlwDTRwABBBBwKeDJoLGwsKChoSFNTU0pIyNDcXFxQY327t2rs88+W9u2bQvaNyEhQWNjY0H70QEBBBBAAAEE1hbwXNCYmZlRQUGBZmdnFRMTo\/HxcbW3tys3N3fVmQ4ODuryyy9XR0eHsrKygtYEQSMoER0QQAABBBAIScBzQaO+vl4DAwPq7+9XdHS0Ghoa1NPTo5GREUVGRi6b9COPPKKrrrpKjz76qP19gkZINUEnBBBAAAEEwibguaCRk5Oj0tJSVVVVWYTh4WGVlJSou7vb3kZZ2g4ePKj7779f09PTuu666wgaYSsbBkIAAQQQQCA0AU8FDXM2IykpSa2trcrPz7cznJycVGZmplpaWlRYWLjirA8dOmTPZ7CjEVpR0AsBBBBAAIFwCXgqaExMTCg7O3tZYFhcXFRiYqKam5tVVFQUtqBx7EA1NTUyv2gIIIAAAgggELqAp4KGOQialpampqYmFRcX21ma2yLp6enq6+tTSkpK2IIGT52EXkT0RAABBBBAYDUBTwUNM4nU1FRVVFSotrbWzsk85lpWVqbR0VFFRUURNKh1BBBAAAEEnkcCngsajY2NamtrU29vr2JjY1VdXa2IiAh1dXVZVnMLJTk5WXl5eUeZOaPxPKo4vhQEEEAAAV8JeC5ozM3NqbKyUuazMUyLj49XZ2enDR2BHY\/y8nLV1dURNHxVykwWAQQQQOD5KOC5oBFANE+bzM\/P26AR7sYHdoVblPEQQAABBPwq4Nmg4fKCETRc6jI2AggggICfBAgaK1xtgoaflgBzRQABBBBwKUDQIGi4rC\/GRgABBBDwuQBBg6Dh8yXA9BFAAAEEXAoQNAgaLuuLsRFAAAEEfC5A0CBo+HwJMH0EEEAAAZcCBA2Chsv6YmwEEEAAAZ8LEDQIGj5fAkwfAQQQQMClAEGDoOGyvhgbAQQQQMDnAgQNgobPlwDTRwABBBBwKUDQIGi4rC\/GRgABBBDwuQBBg6Dh8yXA9BFAAAEEXAoQNAgaLuuLsRFAAAEEfC5A0CBo+HwJMH0EEEAAAZcCBA2Chsv6YmwEEEAAAZ8LEDQIGj5fAkwfAQQQQMClAEGDoOGyvhgbAQQQQMDnAgQNgobPlwDTRwABBBBwKUDQIGi4rC\/GRgABBBDwucCmDxoLCwsaGhrS1NSUMjIyFBcXF\/SSJyQkaGxsLGg\/OoQm0NzcrJqamtA602tNASzDVyBYhs\/SjIRn+Dw3m+WmDhozMzMqKCjQ7OysYmJiND4+rvb2duXm5q5ZEQSN8C0YMxKe4fPEEsvwCYR3JGozfJ6bzXJTB436+noNDAyov79f0dHRamhoUE9Pj0ZGRhQZGblqVWy2ixy+8t\/YSHhuzG2lV2GJZfgEwjsStRk+z81muamDRk5OjkpLS1VVVWUrYHh4WCUlJeru7ra3UVZrm+0ih6\/8NzYSnhtzI2iEzw1Lt5bsXIbXd7N9z9y0QcOczUhKSlJra6vy8\/NtFUxOTiozM1MtLS0qLCxctTLKysrsuQ4aAggggAACz7XAOeeco66uruf6r3X2923aoDExMaHs7Gx1dHQoKyvLAi4uLioxMdEeWioqKnKGysAIIIAAAggg8G+BTRs0zEHQtLQ0NTU1qbi42E52enpa6enp6uvrU0pKCjWAAAIIIIAAAo4FNm3QMG6pqamqqKhQbW2tZTS3Q8xtkdHRUUVFRTmmZXgEEEAAAQQQ2NRBo7GxUW1tbert7VVsbKyqq6sVERGxqe59UcIIIIAAAgg8nwU2ddCYm5tTZWWlBgcH7TWIj49XZ2enDR00BBBAAAEEEHAvsKmDRoDPPG0yPz9vgwYNAQQQQAABBJ47AV8EjeeOk78JAQQQQAABBJYKEDSoBwQQQAABBBBwJkDQcEbLwAgggAACCCBA0KAGEEAAAQQQQMCZAEHDGS0DI4AAAggggABB45gaMJ8eun\/\/fp1wwgn2B6+deuqpVEkQgaeffloHDhzQb37zGyUnJ9tfxx133IqvMo8cmx9u9+ijj9rHjM0PvjvppJMwXiKwHs\/Ay44cOWIf3TaffGt+xg\/t3wLrtRwbG7Pr\/2Uve5nOO+88+7k7tI1Z\/upXv9LDDz+sN7\/5zTzxt84imp2d1Xe\/+129+93v1pYtW9b56udfd4LGkmvy0EMP6eKLL9bLX\/5ymX8QTfv2t7+tV7ziFc+\/K\/c8+oo++MEP6p577tHrXvc6\/frXv7afXRL4NNalX6b5WPjLL79cxjnQ14SNffv28UmtS6BC9Vxqa0LGZz\/7WV1\/\/fW69NJLn0fV8b\/9UtZjedNNN+mrX\/2qfYPxwAMP2H8c77rrLvumgyaFavnkk0\/qwgsvtG88jOH4+Lguuugi3XjjjTCGKPDlL39Zu3fv3jSfYk3Q+M+FN+98LrjgAvsPoPlmYxaLCR3mnc2ePXtCLA\/\/dTPfiM0nrvb399t30nfffbcNGiv9PBmT0GtqajQwMGDD2+9+9zsVFBTYb0DmGxFN9h+2UD0DXgFH8\/8Ejf9W0XosH3nkEb397W9Xe3u7cnNzj\/6k56U\/\/dnP9bkeyx\/84Af68Ic\/rPvuu8\/uWn7\/+9\/XRz7yEf3oRz\/SGWec4WfGoHPfu3evdu3apYMHD9q+m+XHZRA0\/nPpAz\/t9Y477rDbz6aZHydv\/hE0yTwyMjJokfixwzXXXKPHH3\/c\/pRc0xYWFmzg+OQnP2m\/2SxtX\/ziF\/Xb3\/5W3\/jGN47+tnn3WFhYqP\/7v\/\/zI98z5rweT\/Nis\/Nm3j2aUPz1r3\/durOj8W\/W9Vjecsst+trXvqYHH3xQhw8f1sknn6xDhw7Z3Qzz335v67H85je\/qfr6ev3yl7\/UKaecYgPHe97zHpkAcuaZZ\/qdcs35m9tNv\/\/97zUyMmJ\/VAZBY5OVS+AHrv385z\/XaaedZmcXeAceSOabbMphmY75R838JFyzbR9o5tyFeVf4+c9\/fs2\/I\/BOZ2m4C8sX5eFB1utpApq5XWVunZh74QSN\/1789VjW1dXZ2yUmWAS2\/EtKSvT+979fxx9\/vIcrKjxf+noszTm3d77znfYvfv3rX2936c4\/\/3zdeuut4flifDBK4HsjQWOTXew777xTH\/\/4x5clyED4MD8rZdu2bZtsxuGZzvbt23XJJZfoE5\/4xNEBzTdos0W6c+fOFf8Sc1ajubnZvgM3fc1OB+3fAuvx\/PGPf2xvRZlvSuZckdkdImj8t5LWY\/mud73Lvov80Ic+pLe+9a364Q9\/aH8gY1NTk4qLi31fnuuxNGewPvCBD1iztLQ0ewbLfD8w593YHQqtlAgaoTl5rpf5pm0Wh1kUgZ+JYg44mgNQf\/jDH3hXs8oVfcc73qHXvva19r5ioJlbIeYdkDn4eWwzW4NXXHGFPfzZ0NCgc88913O14vILXo9namqqzC9zzsU0s4Nk7M0Ygd9z+bU+38dej+Vll10mc07DvLkIPGli3oW\/8Y1vXFbbz\/c5u\/r61mNpzhj94he\/0L333mufmAjcljY\/TdsEOlpwAYJGcCNP9jCPteXl5dl32YF\/\/My7bnPbpLu725Nzei6+aLMLZOzMjpBpTzzxhN0uNfdpzbugpc08Amu2VN\/73vfq6quv5jT\/ChdoPZ7m3bd5rDXQTDA27xzNP5DmVoDf23osjZdZ6+Yfx0Azh8PNLhE7brK7vaGuc1N\/5naq+f4ZaOZ7QX5+ftDbqX6v2cD8CRqbuBJMajcJ3Jw0\/9vf\/mbfeZt35Sah01YWCOwEmS1m843k5ptv1u23324\/i8AcBDMB5O9\/\/7ve97732R2Mnp4ee6926ecTnH766XrVq14FsaT1eB4Lxq2T5SLrsQzcJv30pz9tn4AyT0hce+219okz8wbE7209lia0mXV+22236U1vepO1NLf0TPAoKiryO2VI8ydohMTkzU6PPfaYPTMQeLTIvKMxzzJvhg9McXlFAs98m79j69at9t52YDfjyiuvtM\/Rf+9737Nb+ubg4rHNfCjNDTfc4PJL9NTYoXoeOyljbs5s8NTJf2XWY2meOll6C9DU7sc+9jFP1Y7LLzZUS3MY1LypCOxymq\/J3JoyT6JwsDa0K0TQCM3Js73M52mYfxjNoSXzTpsWmoD5JLs\/\/\/nPes1rXkMwC41szV54hgHxP0Osx9IcVDZnCszh2ujo6PB9EZtkpPVYmkfdzaOaL3nJS\/heukmu\/0anwedobFSO1yGAAAIIIIBAUAGCRlAiOiCAAAIIIIDARgUIGhuV43UIIIAAAgggEFSAoBGUiA4IIIAAAgggsFEBgsZG5XgdAggggAACCAQVIGgEJaIDAggggAACCGxUgKCxUTlehwACCCCAAAJBBQgaQYnogAACCCCAAAIbFSBobFSO1yGAAAIIIIBAUAGCRlAiOiCAAAIIIIDARgUIGhuV43UIIIAAAgggEFSAoBGUiA4IIIAAAgggsFEBgsZG5XgdAggggAACCAQVIGgEJaIDAggggAACCGxUgKCxUTlehwACCCCAAAJBBQgaQYnogAACCCCAAAIbFSBobFSO1yGAAAIIIIBAUIH\/B3EGCAfqmnf7AAAAAElFTkSuQmCC","height":359,"width":538}}
%---
%[output:442532e7]
%   data: {"dataType":"text","outputData":{"text":"Warning: Could not generate layer plots: Expected input number 2, m, to be integer-valued.\n\nNo layers showed clear bimodal distributions.\nLayer-aware E\/I analysis complete!\n","truncated":false}}
%---
%[output:2d86a759]
%   data: {"dataType":"text","outputData":{"text":"Validating for area: ACA\n  Observed E\/I Ratio: 0.53 (E=9, I=17)\n  Running 5000 permutations...\n  Permutation Test P-value: 1.0000\n  Result is not statistically significant. The observed E\/I ratio could be due to chance.\n","truncated":false}}
%---
%[output:5c8b8866]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAhoAAAFECAYAAABh1Zc8AAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQm8ldP6x5806SqUIhS6IVGUZEr5a5IoU6jIPJQMGW7mpAzVxRUiCRmKQqkkJLpKUooQEa5rKrcylK4S+n9+69517j67vc\/Zzzlr77Wfc37r8+lTnbPed633+zzvWr\/3WVOFTZs2bRImEiABEiABEiABEsgCgQoUGlmgyluSAAmQAAmQAAk4AhQadAQSIAESIAESIIGsEaDQyBpa3pgESIAESIAESIBCgz5AAiRAAiRAAiSQNQIUGllDyxuTAAmQAAmQAAlQaNAHSIAESIAESIAEskaAQiNraHljEiCBXBPYuHGj\/Pbbb1KlShWpWLFirotneSRAAikIUGjQLbJC4Morr5Q1a9YUunedOnXk4IMPls6dO+dtJ4A6Y2uZrbfeWipUqJARm+RrHn30UXn++eelXbt20rt374zuoc2EMsG4qNSmTRs57bTTtLculD8THosXL5YHH3xQvv76a7ngggvkqKOOKlWZiRfPmTNHHnvssbT3O+GEE6RTp04Fvz\/99NMF1xx55JHy+++\/y8knnyzt27ff7PrZs2fL448\/vtnPq1evLs2aNZOePXtKpUqVMnqOn3\/+2ZX1pz\/9SSpXriw\/\/fSTnHvuue7a++67T+D32hTiHsllDh48WL766iv34zvuuENq1KixWbW++eYbeeCBB2TRokXu\/d1vv\/3kkEMOke7du8sWW2yxWf6nn35aZsyY4X7ep08fad68ufZRmb8cEKDQKAdGjvGILVu2lNWrV6csGp3D7bffHqNaRZb5\/fffywEHHODyvP3221KrVq1i65jqmptuukkgNtA433rrrcXeoyQZVq5cKQcddFCRl0JkDBo0qCS3d9dkyuP4448XiA2km2++2XXSodLEiROLFFR\/+ctfXAeHhE6ydevWrnP87rvvZMWKFXL99dfL2WefvVl1xo8fL9dcc03aakKkPfLIIxmJzQ4dOshnn30mI0eOlI4dO0qibSBodt55ZzWOEPdILNSz8T\/D+4f3MDEtXLhQzjzzTFm3bt1m9cXHwV\/\/+lepVq1awe8gyA8\/\/HAnMJEg6oYMGaJ+Vl5Q9glQaJR9G0d5Qi800HCh8cXXERr31157zdXn9ddfl3r16rl\/\/\/HHH7J8+XLZZpttBF+U6RLC4vh6rFmzZqEs+Bm+JKtWrVpwP4ic2rVrZ9RR+JutWrVKDjzwQPffZKHx66+\/Cn5ft27dQl92qa75xz\/+If\/617\/cl+yf\/\/znQnXFz9euXSv169d34f106YcffnBfnOm+qjds2CBz584tuPySSy5xHUSPHj0KvuDBd4899hDUHR0v6g5OqdK3337ruCZ2JEXx8PdAPRo3buz+O27cOMcv8cs30+dFx5rKXolC46GHHtqs6rvttps0aNDA\/fz+++93neFtt90mw4cPz0hobLXVVi4ag\/Tvf\/9bnnrqKXnllVfc\/5999tlCX+gQXrAHol2JKVlogPc777zjsiA64v0S\/4ftwRrioyhfL+oevmwwyzRagmcEF58gyCCGfUJ5iETBd7fbbju55ZZb3HM+99xzMmHCBJctWZzgGU888cRCLJYsWVLIh9I6OH9RrghQaJQrc+fuYb3QQLj21FNPdQV\/\/vnnBZ0gwuGHHXaYEx\/4CvZfUS1atHCdxE477VTwhYrOAMMEaOggXNB5X3HFFYKvTnTWvmPo27evNGzYUG644QZ3P3S0o0ePlj333NN1INdee6372p00aZKrD+owcOBA9xWMKESXLl0K6oHGdujQoa6jwf1eeOEFdw3qgs4U1yFcnuqaefPmuXITIwrz58+Xfv36uc7Pp\/PPP18uv\/xy9wzghC9ofBV+\/PHHLkKAshAtwFd5UaIE92vatKmr+4033ihnnHGGK+KXX35xzHBfn7p16+bK8p0fOhvYYOnSpS4Lng31hChJ9Wxt27YtuNcXX3whJ510UkHkCswuvfRS99zFPe\/VV1\/tOrALL7xQ3nrrLcHXdKpOygsN3HvBggVpHTjx6\/rdd991wymZRDSS74vIBIQDko9QTJ8+XQYMGFDwnPAx2Om8886TY445Rj788MMC34B\/gr8XrD6igYgCbJ34DPBfCKNUYiFR5Pl7IKoAO2F4atq0aS6SAB\/HMA2GjIpKXgzhncPQEhL8dPvtt3f\/fvnllwuG+Z588slC0bKLLrrI+T\/ugWEVnxAtGzNmjOCdhf2QRowYEXToLHctFkvKJgEKjWzSLcf3TiU0pk6d6joiJHT2iDr4sWw03ugY0Fmik5s1a5b78kvs2HDdxRdf7ERIUWHvROz+y23s2LFOMOy9995u\/gQSOlkIDHQKGLOGEEGjjoTr0OG++uqrrvFEQseABhV1xLg\/OvVU10yZMqXQ0Am+EjFfwyd0Dj7cjDkc\/fv3dx3ZE0884bLg+RG98UNP+BI95ZRTivSmVEIDQsl3DPvuu6+899577h7HHXec3Hnnne6rG1+kEDRdu3Z14gAdLf4PRqhTMo\/EMXh0nuDnhR6YIaLSqFGjYp8XwhEiIjEVJTSQL7kzhbiACEOUBnaB6PHPduihh2YsNOBrSIhoQHx6e7\/xxhuOBaISSPDRbbfdtqBThZ0ffvhh99WPBN\/q1auX81k\/rAV+6MwhfOAHSHvttVeBsIPwBYfk+UCphk7gf95vUBfYyifYIDl65n\/30UcfydFHH+3+i\/kUYAQfhlj2TH00qDhB5+8J\/9x\/\/\/3dfYYNG+bekxdffHEzMVKk0\/KX5YYAhUa5MXVuH9QLDXSqO+64oxt28A0tOj00zpjDgM4NY\/qIamC4AF9HSKNGjZLdd9+9QGigoUTHD5GBBh5CA50AGk6sLvANO8ad0TEjooCvRd9wFic0EPFINVQAMQHBA0EEQYJGFZ03ngvDP6muSZ6jgQYdHRiugdhCSBpRm7vvvts96\/vvv++iJxAayPPSSy+5n6PjxNcyyoagKSolC43EIQ2UhejE3\/\/+dznrrLPcbRBFQDgdnNBpgRWGIPAVj4RoCyI26YaSfF0QNdlnn33cf2fOnOnukcnzgqsXGrA1fMJ\/XSc+Z3FzNLw4AR\/Y0EfKMhUa6Zj6uR0QY5jQibphOCHR3p5rcXM0EGGBQEZCJAJDTYgq+E4eQ06YJJ2YihIaXkxhuLFVq1YF70uqSa\/4JXwWdoXAQWTCR5P8e4g8\/md4\/zDBs7iU6EsYZsS7gIgNEphhGJSJBDwBCg36QlYIpJsMio4I4Xx8GfvOEWLAj3t7MXLZZZe5ztFHNHwDjcr6iXy+4cTP\/Neeb\/zxhYUOGmIEHXlJhQbG1BEeRkOKsLUf4tEIDQgGfHGjk0dUBQnDI351BoQThhEgNBInkPqvfnRI6Lw1QiNxCAB19XMzPF90bpgbk7wqBUMrmFwJO2UyRyOV0MjkecEUIgLDAffee2\/aR0sUGojCJCbMBYEAXb9+vfMl+BGGAyA8NULDz\/HwbFAGhieuuuoqVxzmwjzzzDNuOCsxT6ZCA7a95557nKDzKzTAHtGM5OEu\/3xFCY277rrLRaCQ\/HuG+\/uoRSIjiEWIGETH4EcY1kGUxfuTF4eYtOxFp69jUf4GUYGPBURxUPaPP\/5YMF8Dohk+wEQCFBr0gawS8A0gxuB9A4jJfokT\/rzQwJdVctgXX4n48vNCI\/EryQuNxGEQDE2gE0Cnhc4rE6GBL1WIHny1p4toINqCzgudGCa1IlyPvBqh4VdlIEqAL0ekRCGQKDQSoxcYUkEHVxKhkRguB0sIrsSEIRvMXUHIHcs8\/RCJz4NOFVGRkkQ0MnleLzQSmaRyyEzmaIAfol2YEIu\/kTIVGolDBRgOQDQKIgy8MNSEyI9fRYMOG0s9ETVBx52p0IC\/YDgm0V\/hR2CL+6RaGVOU0EAEyEcvihMa8N2iVgFBMGAOBp4Z9UDCMyfOG8E7BbGF9xFCGZOv8c6mS\/59ymoDw5ubIsCIhilz2alsqjkaybX3X76IXKDRxlcevv4wAx6NOiZAeqGBjs+v+y+J0PCT3dCBvPnmm27PA0zcQwg7ldBAY4vOwIe0\/QoEP5cildDwDXTy0Ml1110nmGCHL2fMfcDKDoSyEdJGQmgdggcRjVBCIzHSgK9VREqwasLPT0H4Has4MM8CwgwdKIax\/NAK7IGfeaGR3Pl4W6aKaGTyvJhICBGRGDkoSmjAbomrbHxeRDUQgYEd\/dd5SYUGroPwgvhBwpAAhrcgLPxcHwyjYUIlUrLQ8BMhk0UC9qTwc5MwH2SXXXYpmB+D+\/jhnsTnDyU0vC3ADyudfML9IXK8H2PuB6KCSH4pK6Jg4IGPBSS\/lDhxrhWiij5BgPg5JLAHhjmZSAAEKDToB1khkInQwIRQrB5Bwlc3lmD6CYvo2BBWDiU0Ele8oNHFhEs\/mc4LDSzB9RP\/ECHBlx46fjTI6JghdPwmT7geHV\/yNehQIEoS99H45JNPCjaVQtkIofvnhNiBePECJpTQAFMfEUGZ4IgVD5hwi69RsMfXqf96R9Rpyy23LNgcC2PwGGdP5IFna9KkSSF\/SSU0MnlePyyUqdBI56QYgkO0C\/4GAepTSSIauBZ+hwmtSIjy4A\/mfyDygWEHdLzedn65J3wDP0PEAs8DcZo4GRRRvCOOOKJgxRF+j0gDEjpqdNzJu5iGEBqwDXwb7xGGIv08EZSbKKjgCxjGwYRoPxEWPgN\/8BOSIUgwfIl34JxzznHL1PEc8B+fEvddgcBBPiYSoNCgD2SNgBcaxW3g5Ge7+4qgQceXIjoKLJ9MJTQwWQ3j56mGTvxXZfLQCe7\/t7\/9zY0nI6EcbDaEr+rEUK+fFIc8yIuvtMTlt2g8\/X4OmLSJfSqSr8GXMIQGOixMIERCfdBRJ26GhCgDOntEOPwEysShBM3QieeduLwV80sQXUlc3YEOBXzxVY3Jt1jq6ldNeBskrkZIfrbkeQCJQsN\/rWfyvP6+pRUa+ErHbpfJezx4oZHII9HZMSQFvsmrLBJ9DhwgIiAw\/IZkiGZgSAmizUfiEjf\/wjAXlll7oeG\/7CFq0dH7ZcSoCyZeYr5Fqg29EoWGv4cfHsRcCv9eFDV0kigmvK96Bol28zZABA9COnk+EKI58BP4TKKY8JGyRK5+qDHx3cxaI8MbmyFgPqKBLzSMRycmhOywxA4JjQK+HqDM8VImhg\/NWKmMVxSNHhp4fEGhMcvmGRXwB0xy3GGHHdJuhoUOGhMM8UWP4Rtcg5DwrrvumvE1qUyGMzjQKSIKgmGU5I2fsmVmCArUH+PuiMQkJ9Rn2bJlbgOpVBtJJfPItJ6xnjfT+mnyYcgEQwnpNsiCD0OUgmHipmfJZaC9wmoRRAgy3WxLU88Qeb3dICrQXqZaDRSiHN6j\/BAwLzT8XgiJJvNfkmhAsX4dDSVeFkwWTPwaKD9m5pOSAAmQAAmQQBwC5oUGQqOYEIa\/kxPGvRHO9WOLCP9hsiGWGha302Icc7BUEiABEiABEihbBMwLDYyfYuwSexJgjDExzIdZ1Ihu+EOXMKse4+IYU8UwChMJkAAJkAAJkEB2CZgXGn7Sl8eESUhYkoaxZqz79ucV4Pd+Jz2\/10J20fLuJEACJEACJEACpoUGJi1h9jVmwmOHQyyrw7IqLM3ChkCYLZ24Rh0b8mCSqF\/\/ns78fuY03YMESCB7BLquWyddf\/65oIAp1avLlKSNxbJXOu9MAvlLIHnpcP7WNLOamRYaqR4RhxxhOSKWcx155JFuUyRsq4zkz9LAToLJ+wEk3gu7VGLfhRAp5L1C1Cff70FeOgtZ5IU6Y0ntttOny7b\/PRUXT\/1j587y41FHuSWlod6\/ZJoWeek8IlxustKxDMkr5L10T5Gd3KaFBhojbHaDTY\/8IT7+TAts4oM17zhNEevlkfx2vKlOiaTQyI6Dae9a1l4w7fNr81vkRaGhtXKc\/BZ9Kw6p\/5QaklfIe8Vk4ss2LTSwdh0TPjFEgo2RIC6wARL2SMCOdf6Y7MmTJ7v9A7DTI\/ZoSNzNLpURQho55L3ywWGyXQfy0hG2yItCQ2fjWLkt+lYsVhQaRZM3LTTwaDj2eMiQIQV77GN7ZUz2xIY4ECI4PMofGIVNkhDxSLVpUbYiGtiZMdXS25gvRD6XTV4661jkFVNoWOSl84hwuclKxzIkr7Im8swLDe8KGEbBrnypdrHDahPs9OiPgy7OfUIaGZuEZVpucfUqD78nL52VLfKKKTQs8tJ5RLjcZKVjGZJXyD5I9xTZyV1mhEZIPCGNHNL5Qj5jvt6LvHSWsciLQkNn41i5LfpWLFYoNySvkH1QTCa+bAqNFFYIaeSQzpcPDpPtOpCXjrBFXhQaOhvHym3Rt2KxotAomjyFBoVGzHdzs7LZuOnMYZEXhYbOxrFyW\/StWKwoNCg01L7HiIYaWbAL2LjpUFrkRaGhs3Gs3BZ9KxYrCg0KDbXvUWiokQW7gI2bDqU1Xtx1V2df5i47BDS7fYbsg\/KBIIdOOHSSD35YUAdrHWdseNZ4lbUGNLb9Wb4dAhrf1+S1QIBCg0Ijr\/zUWscZG541XmWtAY1tf5Zvh4DG9zV5LRCg0KDQyCs\/tdZxxoZnjVdZa0Bj25\/l2yGg8X1NXgsEKDQoNPLKT611nLHhWeOV3IDm+5wNzbh6bF9g+flNQCMeNHnz+6n\/UzsKDQqNvPJTax1nbHjWeCU3oH4FSmyO6crP5kmy+fDMGzdulMqVK+dDVQrqsGnTJlm1apXUqVOn4Gc4TuLHH3+UHXfcsVBdS1P\/n376qeAwzjVr1sif\/vQnqVSpkprFH3\/8IfiDhPvUqlUr5T004kGTV13hCBdQaFBoRHC79EVa6zhjw7PGy7rQ+OKLL6Rt27ZywgknyO23315g\/qeeekrw57nnnkvrEr\/99pvsueee8sorr8jq1avlwgsvlAULFhTKv3LlSjnooIMK\/QznN5199tnStWtX9\/O77rrLne2UWH5yoS+++KLstddesttuu21WH3\/95Zdf7k64\/uijj6Rq1aoZu\/I333wjixcvls6dO8ucOXPksssu2+w5Mr5ZiowTJ06U9957TwYOHCgoa9iwYe6UbiQc59CtWzfp06ePO8tqn332kb\/\/\/e9Sv379jIuEwDj33HNl6dKl8v7778uVV14pKHP69Onu3o899pg0b9484\/vh\/KzXX39d7rnnHoEwHT16tOy0006bXa8RD5q8GVc0YkYKDQqNiO63edHWOs7Y8KzxKitCA3aHsDjwwAOdC2iExowZM9wXOzo7dHSJyQuNJ554QvbYYw8nKJ555hl58sknXYeLjvCrr76SX3\/9VRo2bJjW\/SACcFo1\/k5O\/vpq1aqVSGhAKA0aNMh1rui0cc6UpmMu6p3B\/SB+Xn75ZRe9uPrqq+Xnn392B1NuueWWAnYQBo888ohjXxKhMX\/+fOnevbssWbLEneYNQTZp0iTZb7\/9pGnTpiUWGg888IATLBB5o0aNotBIIEChQaERu68sVL61jjM2PGu8yorQOPnkk91XPL6Cq1SpUkho4IsYIXR09EiPPvqo\/Pvf\/5bzzjvPRTQyERrTpk2Txo0bF7jXHXfcIc8++6zMnTtXxo8fL\/\/617\/k4osvdp3amDFjXKfZoUMHFwV48MEHZcSIEe6UaogTHCr55ZdfynfffeeGBtCh4voTTzzRdeq4z7hx42Trrbd2nfhRRx0lCxculJEjR7p7IeFZ0blfc801cuqppzoB1KlTJ7ngggvkvvvuE3Syv\/\/+u9x\/\/\/3uXhjSQAQG99tiiy3klFNOkR49eriv\/Q0bNrhoDjr75AR2b7zxhrsfUsuWLeW0006TSy+9tCAr8kBg7L333u7vSy65RMAL98W\/IcbS1X\/AgAHSq1cv+eyzz6RNmzYukgN7IGr0t7\/9zdXZRzTeffddue2225yQat26tdxwww1Ss2ZNN0xy9913uyjLzjvv7A7yXLt2rasz6gC7vfTSS04oJiZNlEKTN3YblEn5FBoUGpn4Sc7yWOs4cwYmTUHWeJUVoYFwPTrPs846ywmIxIjG0KFD5YcffpAhQ4Y4q+H\/69atE3RyJRUaEBjocNH5PfTQQy6qcf3118sBBxwg9957rxMJt956qxx\/\/PHSrl07OfPMM52QQKeK0D46UXSWEAZvvfWWux4iAEIDnSw6fkQR8EWOSMWyZcvkuuuuc8IG6dVXX3UiBh0oBIf\/g3kTfugEDG655Rb5y1\/+4qIE\/fr1c3XAvSEIMOyBe0AUTJgwIeWQDSIYO+ywg7snEv6PvF26dJHDDz\/cPe8uu+zifueHTnBfPAueCyIBQ0God6r6Q1R4HsgL0XDSSSfJ8OHDpX379i5Kgp\/Xq1fPDWGB39FHH+1EBCIrEHkQGNdee617PkSW\/vrXvzqR58UR+J9xxhly+umnU2j8lwCFBoVG7L6yUPnWOs7Y8KzxKitCA+H3RYsWSe\/evWX27Nnuj5+jkQ2h8cEHH7ivbXSmGFaBULjiiiuceMCwAjp0DDvgixqME4dOMHcAnSeuRXQBoiNRaOB3EBzodBHtgIDBV3qqjhoiJHHoJHGOBuYnoD6oFxJ44Mt\/5syZTmig3oceeqjrsCFu0OknD\/\/gHhBUiEog4XnQuSNyM2\/ePPcz1BXCAEMpuK+vP6IKqD\/uiwhOuvpDhCBCgiiNj0DMmjXLCRg\/dPLOO++46AuiKxUqVHBRDQgRMESUY7vttnP3R4J4Q\/JCA\/+vXbu2E12JSROl0OSN3QZlUj6FBoVGJn6SszzWOs6cgUlTkDVeZUlooDM555xzXPgdYfh0QmPw4MFuKKE0EQ1EG9CxoXP0QuHOO+90cwF85OSII45wIgFf+MlC49NPP3WdM1Ky0MA8ka222sr9Dl\/hEAONGjUq1FGj\/JtvvtlFO9IJDXTSGOLp2LGjuxdECO6HYR0IAggO1A0JfpA8POR\/jqEmCBYMxUCUbLPNNu4aRDAgIsABwy6YzJp4X+THcAXuu2LFirT1z0RoTJ482QmY5OQnjII5Ih1IGF56++23C4QG7I05OJ63v4dGPGjyxm6DMimfQoNCIxM\/yVkeax1nzsBQaERBnby81a86QUQDQgP+ilA5OkZEFLDqBBENTOr0q0LOP\/98N1+iNEID8z0Qpoew8EIBcwYwdIEVDpiTgI4N5eBvjdCAIPCrJDAnAhEQdOr9+\/cvWE2Czh9DNkUJDXT+KNcPGSCCAWGAuR7JkzbTCQ2IFUw0xRCQjyL4aIN3AAynYNUOoiWJ900WGunqn4nQwFARolSYhIuEFUNYpYKICeaoYB4LVgIhoT4YKvMRDQggzN3wkR0KDe6jkbLxCqkm2XHq+gfyKtu8ylpEA9ZCx44\/GA6A0MCXMMLumAsA8YEhDwwHaIQGJlhiPgfmdiBSgg5vypQp0qRJkwKhgY4OEyzxBY+wPwQIhlggCI455hj3O3SKEA5FRTQghDAnAve\/6qqr3BwHdPL4YkfZu+++u5uPgmEGCA10wsj35ptvuuEMP0fj4YcfdveAAIDgwfyGI4880tUjU6EBkYIVLLgnokCIruAPogSYh\/LJJ5+41TpYXowhinRCA3ZJV\/9MhAYiKZhngeeH+IHAg3DCM0N0YZUKuKKOYLz\/\/vsXCA2IJMzd8cuRKTQoNCg0dP1a1nNTaOgQW+NlfWfQ5IiGD+kjqoF5DRAa2PsBEwwRvkeHiz0eMBThhYbfRwNCId3y1kQvwJAIhAr+RoKgQBmImCDS8cILL7g5A+iIscqkRYsW7m9EEjBRFKIBfzDUkni930cDEyARoUHCZE109viCR\/0Q7cCwyiGHHOK+6CE08FyYnInybrrppgKhgSgOOmfkQ4LwwrBC9erVMxYa6Lw\/\/vhjV28kzJXAfAqscvHDOxAvmP+AOSXphAaGUNLVH+IIzw7BgSgRJq76vTgSl7eiDIg2JPCFmITowYodiAusXEGCXTB8BnHo57lAGEIUJibNB6wmr67FiJObQycpuIc0srWOII4b\/q9U8tJZwBqvkO+WjlRuc6PDwXJSCA1MJsxmguhAh+nnP\/iyEM6HGMBeEcUl3ANzISAKEhPEA36OJbyJCUJk\/fr1m+XH8AUmYiI\/hmO0z456YNksBA2WkiKhLExeRfQA0RVMaM00pat\/ptdjrgWGabDxWeKmZqgTnhPiMpEZhBlW\/zz\/\/POb1VPj+5q8mT5LzHwUGhQaMf1vs7KtdZyx4VnjVdYa0Nj2L4vlIxqD7cD9PiSWnhHRIAxFYWVMctL4viavBT4UGhQaeeWn1jrO2PCs8SprDWhs+7N8OwQ0vq\/Ja4EAhQaFRl75qbWOMzY8a7zKWgMa2\/4s3w4Bje9r8logQKFBoZFXfmqt44wNzxqvstSAYl4EJilioqDf68H7Q2lOFc2FT2GOQUlOKs1F3cpqGRrf1+S1wItCg0Ijr\/zUWscZG541XmWhAcXSR2wa5U8UhQ9gy2psVoV9KL799tsSHVaWS1\/C6gps7Z14ngrKxxwDTGhMTFg94rcrL6qO2T7VNbFsrKLB0lOsAsJW4Vjtkko44eyVf\/7znwWXwv8whwJH0T\/++ONurwwIRWxvjr0xspk0vq\/Jm806h7o3hQaFRihfCnIfax1nkIcuxU2s8bLegKKDwoFq2NAKZ1xgaSS2vkaHhg2bsL9CnTp1zAqNnj17uiWj6IyRsLIEG5BhCS12+MSJr+lSNk91TSwTe5NAFGAFCvbowAFyEHjYij054ecQIbVq1XK\/wioRbCWOvU6wlBbn1GB1C\/YF8Vukl+J1LPJSje9r8marviHvS6FBoRHSn0p9L2sdZ6kfuJQ3sMZrswYUx6T\/d\/fFUqLIzuVNm4r06FFwb7+tNs7ewIZaPnkBguWsOOE03amoyA9RgqPfIVbQsfft29ctA013WujTTz9dcPoqNs3CFzpWZvjysZkVlpJiO3TcFxtnIeqCjaNwMiu+9LE3Bfa8wJJbbOaFTjbVFuCoD\/bVwIFhPmHfCfwcX\/\/Y8RLyrCEXAAAgAElEQVTbkWN\/CX+qKQ5Rw1CM9lTXokRLUcaEmIPI8we+QeBAGCULIX\/oGjYgS1yaintDYGBvEL+7JyI5EI04LC1bSSMeNHmzVd+Q96XQoNAI6U+lvpe1jrPUD1zKG1jjlVJoZLFxLyVeEQiNW28tuI3fIRL7PCQnbDKFMzKwc2S6U1GxrwM6RdwH8zvQsSGEjw483WmhEAWJp6\/iyx0iAqeiYudQDIPgaHZ0rBAb+D0OK8MW3NjRExte4SwW7LOB\/+PkVWxZnk5oYKMpHBaHBGGCiMF7773n9rbAvJRWrVo5AYPtuDFchMgCxIb2VNfkLbohhiCQkhPYQMD5dNddd7mt3\/1ZIrgOEY7Es1SQFyfQYnMvbKi1fPlywXby2EgNEQzs94H9ODD0AhZg5Q9nK7XPpLmBRjxo8marviHvS6FBoRHSn0p9L2sdZ6kfuJQ3sMbLutDAYVr4wscOoMkJkQqIBkQVIDRSnYpao0YNtz8EwvSIHKAzxBwBbPCU7rRQzEVIPH0V5UAk4Bp\/2BrqBHGAe\/lD1nDqKQQN7oudS3HwF4YQ\/Emw6YSGPyU18fnQqWM3UGwCht+jY8dmVhBXiCQg6qI91dVHJHw5EEl+h9LEsnGWCKIlPmFXT2xEhl1MkTCUgm3Lk58HW6Vju3JcD3GBnVLBf+LEiQWbaeFesCV2HU01Z6WUr2ehyzXiQZM3ZB2zdS8KDQqNbPlWie5rreMs0UMGvMgaL+tCA9EKDEFgmAOdXWLCXAAMbWCrbwiNVKeiImSP+QDo7NC54Th0fE1jHkS600IhJhLPKvEHuWEYB9uQ77DDDu6eEBP4XWJCGTjRFYIDX\/xIiYePJU8G9XM0\/NHna9ascUNBiAggooHng7gYM2aMi6Yg0oCIQ1FCI92prhh6KUnCRFyIC79NObZEx9bgyTbByh9EebydsM04ziBJjnxAMEFwIFoDUZetpBEPmrzZqm\/I+1JoUGiE9KdS38tax1nqBy7lDazxStmAYp5GPicMn\/w34Zh2DD+g48akQp\/8fAB0WDjwC0Ij1amoOPcE23PjzxtvvOGGOTCEgBUb6U4LxSTTRKGBMjEMgOETiB5\/2Br+f\/DBB8sll1ziqoWTXdGJYrIqzkrBgWSYr4GycNpspnM0fGQAcx1ee+01N2SDIaK9997bHRyH1TdFCY10p7pCtCUm3AvDScmpbdu2brKnTxAYmCODiA4SbII64e\/EhHNSIB4wJIWE58f25oiaQPBBQCGqhISyMUSUHGUJ6ZYa8aDJG7KO2boXhUYKsiGNbK0jyJajZXpf8sqU1H\/yWeMV8t3SkQqXG0IDZ29g7gTmM6Azx9Hm6KTw1Q\/RAaGR6lRUdGj4ssYJpwjjY9UEjlYHl3SnhWKoJFlo+MgK5l34SAXmS2B1CCZqIpKBL39MgsTPcdAalqhixQzKhnjJVGj4oRbMZcCwEIYYpk+f7oZRECGoXLmyW22jPdXVT8T0lsFzJkdk8DsM2UAY+QTBg5NZwQBzUfzQCIZ3IKJeeuklJ6xgC9gA9cI+JzgkDQeiIXKEKBLqD6GHuRqYxwKRljxvJJzXiLNxplEcTd6QdczWvSg0KDSy5Vsluq+1jrNEDxnwImu8ykIDirA9vobxVe0Tvu5xeieOa\/f7aKQ6FRWTQY899lh3AirEAFY6IDqCOQTpTgv1J7D601dRph8uwCTMPn36uGpgmAPzNPwcC+x\/ga90DG14YYJ8EB04YTXVnIRUq04gqrDPBDpyRBYgVtBJI6FDh2i5+eabBZEHzamuWAZc0gRRgpUnSBBbmI9Su3ZtF0XCCpLFixc7vojuQHwhYf4K5mng+SGeIJJwYBoSxB5sClbZShrf1+TNVn1D3pdCg0IjpD+V+l7WOs5SP3Apb2CNV1lqQLFCAqsW0Dn5k0aTzZnqVFTMc8DJn9tuu63bcyMxpTstNFM3wTLbxNNcE09P\/f77750Y2XXXXdWnqiaWjzLw3BgGwlAMhmbwN5arZutU11TPj3LBK\/nU2uS8yAdhhCXAiZt6+Z1dIVCS59tkyluTT+P7mryaOsTKS6FBoRHL91KWa63jjA3PGq+y1oDGtj\/Lt0NA4\/uavBYIUGhQaOSVn1rrOGPDs8arrDWgse3P8u0Q0Pi+Jq8FAhQaFBp55afWOs7Y8KzxKmsNaGz7s3w7BDS+r8lrgQCFBoVGXvmptY4zNjxrvDDZMNWGULE5snwSyDYBrGrBDq6ZJAqNTCgZzxPSyNY6gtimIy+dBSzywvuF3Ri3nT5dtv3vigA89Y+dO8uPRx3l9ojIdBmgjpa95cDa5wuZ36JvhXx+7b1C8grZB2mfIxv5GdFgRCMbflXie4Z8WUtcCUMXWuRFoWHDwSz6VkyyIXlRaMS0ZI7KDmnkkM6Xo8ePWgx56fBb5EWhobNxrNwWfSsWK5QbklfIPigmE192mYpoYMc4bEOLDVt8wpp1jAljYxZsNoO138WlkEYO6XzF1bss\/J68dFa0yItCQ2fjWLkt+lYsVhQaRZMvM0IDG9Ecc8wxbmMWHGaEhJ9hb3ts2LL99ts7xYmTDLGDXVGJQiPe68rGTcfeIi8KDZ2NY+W26FuxWFFolBOhgX38\/YmIXmgMGDBAZs2a5fb0x7kCgwcPdtvuYs9+HGqULlFoxHtd2bjp2FvkRaGhs3Gs3BZ9KxYrCo1yIDRweiFEBA4lwr77Xmi0adPGHVrkzwLAqX04SRD74ieeBpiMiEIj3uvKxk3H3iIvCg2djWPltuhbsVhRaJRxoYEzA3CSHw40wmFGONwHQgNzMxo3buwO0enYsaOjsHz5cmnVqpXgkCIcosOIRszXMnXZbNx0NrHIi0JDZ+NYuS36VixWFBplWGhs3LjRnSS4\/\/77yw033OAiFV5o4GAhHC2MI4FxZDMS8jdq1MidQohTBosSGsm\/Q7QkcZJppg799ddfS7169TLNXu7zkZfOBSzyateunTz++ONSZ8YMqT1jRsEDr+rQQVZ26CC9evUqOPpcR6P43BZ5Ff9U2clBVjqupeGFfgon7CambO0lo3uqMLlNTwaFsMDRvldffbVsueWWbsUJ5mMMHDhQDjnkEDnyyCNl2LBh0q1bN0cLE0VxRDCGWpo0acKIRhgfCnoXfkXpcFrkxYiGzsaxclv0rVisGNEowxGN6dOny6RJkwqeEIpy6dKl0r59e7nkkkvc\/Ax8HfXv39\/lwTJXbIG8ZMkSd6RxURGNUGqSL6vu1Sevss+LQkNn41i5+S7qyIfkFXKeoO4pspPbdEQjGQlEB1aa+MmgQ4cOdZNDJ0+eLHXr1pWLLrpIKlasWOx+8yGNHNL5suMC+XVX8tLZwyIvCg2djWPltuhbsVgxolGGIxrJj4blrTfeeGOB0Pjll1+kd+\/eMnv2bJe1QYMGMnbsWCc6ikoUGvFeVzZuOvYWeVFo6GwcK7dF34rFikKjHAmNdI+K1Sbr1693QiOTRKGRCaXs5GHjpuNqkReFhs7GsXJb9K1YrCg0KDTUvkehoUYW7AI2bjqUFnlRaOhsHCu3Rd+KxYpCg0JD7XsUGmpkwS5g46ZDaZEXhYbOxrFyW\/StWKwoNCg01L5HoaFGFuwCNm46lBZ5UWjobBwrt0XfisWKQoNCQ+17FBpqZMEuYOOmQ2mRF4WGzsaxclv0rVisKDQoNNS+R6GhRhbsAjZuOpQWeVFo6GwcK7dF34rFikKDQkPtexQaamTBLmDjpkNpkReFhs7GsXJb9K1YrCg0KDTUvkehoUYW7AI2bjqUFnlRaOhsHCu3Rd+KxYpCg0JD7XsUGmpkwS5g46ZDaZEXhYbOxrFyW\/StWKwoNCg01L5HoaFGFuwCNm46lBZ5UWjobBwrt0XfisWKQoNCQ+17FBpqZMEuYOOmQ2mRF4WGzsaxclv0rVisKDQoNNS+R6GhRhbsAjZuOpQWeVFo6GwcK7dF34rFikKDQkPtexQaamTBLmDjpkNpkReFhs7GsXJb9K1YrCg0jAiNadOmyR577CF77rlnTF9xZVNoxDMBGzcde4u8KDR0No6V26JvxWJFoWFEaJxxxhnuOPe9995bTjjhBOncuXOxx7lny6koNLJFtvj7snErnlFiDou8KDR0No6V26JvxWJFoWFEaKxcuVJmzpwpU6ZMkXnz5rlat27dWrp16ybt27eXatWq5cyHKDRyhnqzgti46dhb5EWhobNxrNwWfSsWKwoNI0IjsZoQHa+++qpgOGXOnDmy1VZbyUknnSSnnXaaG9bIdqLQyDbh9Pdn46Zjb5EXhYbOxrFyW\/StWKwoNIwKDUQ3XnjhBSc0kLbbbjtZvXq1XHLJJdKvX7+s+hOFRlbxFnlzNm469hZ5UWjobBwrt0XfisWKQsOI0FixYoUbOpk6darMnz\/f1frAAw+UE088UTp27OiiGuPHj5cbbrhBPv\/886z6E4VGVvFSaATEa7EzoNAI6ABZvJVF38oijmJvHZJXyD6o2IrnIEOFTZs2bcpBOcUW4SeDNmzY0A2THH300bLzzjsXuu6bb76R4447ThYsWFDs\/UqTIaSRQzpfaZ7JyrXkpbOURV4UGjobx8pt0bdisWJEw0hE47HHHpNGjRrJQQcdtFmNv\/rqK9lpp52kYsWKOfEjCo2cYE5ZCBs3HXuLvCg0dDaOlduib8ViRaFhRGggonH44YfL2WefXajGcPZ27drJSy+95PbZyEWi0MgF5dRlsHHTsbfIi0JDZ+NYuS36VixWFBp5LjQwFPLZZ5\/JunXrXE0xFyMx+Z\/PnTs3Z\/tqUGjEe13ZuOnYW+RFoaGzcazcFn0rFisKjTwXGqNGjZK1a9fKs88+K7vssou0bNmyUI2rVq0qhxxyiLRo0SJnPkShkTPUmxXExk3H3iIvCg2djWPltuhbsVhRaOS50PDVGzt2rNt+PFloxHAcCo0Y1P9TJhs3HXuLvCg0dDaOlduib8ViFbrtCtkHxWTiy4666mThwoXy6aefut0\/MTTy7bffpmVy\/PHHS5UqVXLCLKSR+bLqTEZeZZ8XhYbOxrFy813UkQ\/JK2QfpHuK7OSOKjQGDRokY8aMkaVLl8rFF18sM2bMSPuUECU1a9bMDoWku4Y0ckjny8nDRy6EvHQGsMiLQkNn41i5LfpWLFaMaBRNPqrQ2LBhg\/z2229uAij+vXHjxrS1rV69es58iEIjZ6g3K4iNm469RV4UGjobx8pt0bdisaLQyGOhkVy1L7\/8Ur777js3T+OLL74Q7K2BDbyOPfZYodCI+Qrlrmw2bjrWFnlRaOhsHCu3Rd+KxYpCw4jQ+OCDD6Rr165y+umnu\/NMDjjgABfpwPLWBg0ayIsvviiVK1fOiR8xopETzCkLYeOmY2+RF4WGzsaxclv0rVisKDSMCA0IDAyjPPLII+7U1iuvvFJeeeUV97NOnTrJ008\/nbMlrhQa8V5XNm469hZ5UWjobBwrt0XfisWKQsOI0GjatKlcffXVcuqppzqRsXjxYjc5FEIDy15HjBghRx11VE78iEIjJ5gZ0QiA2WJnQKERwPA5uIVF38oBlrRFhOQVsg+KycSXHXUyaCIArDrBUfDYihxDKPgbggPHxCPagZNbc7XHRkgjh3S+fHCYbNeBvHSELfKi0NDZOFZui74VixUjGkYiGhgmOf\/88wtqi2jGsmXL5C9\/+Ytb1or\/Y5fQXCQKjVxQTl0GGzcde4u8KDR0No6V26JvxWJFoWFEaKCaOAYe8zOaN2\/uohcvvPCCOwcFwym1atXKmQ9RaOQM9WYFsXHTsbfIi0JDZ+NYuS36VixWFBqGhEZMJ0ksm0IjniXYuOnYW+RFoaGzcazcFn0rFisKDSNC4+eff5a7775b3nvvPVm5cuVmtZ40aZJsvfXWOfEjCo2cYE5ZCBs3HXuLvCg0dDaOlduib8ViRaFhRGj0799fnnnmGTn44IMFDVGlSpUK1fyqq66SatWq5cSPKDRygplCIwBmi50BhUYAw+fgFhZ9KwdY0hYRklfIPigmE1923qw6adOmjeyxxx7y0EMPRecS0sghnS86mBxUgLx0kC3yotDQ2ThWbou+FYsVIxpGIhrdu3eXVq1aucPVYicKjXgWYOOmY2+RF4WGzsaxclv0rVisKDSMCI3p06cLhk+eeOIJ2W+\/\/WL6ixu6+fzzz4PUgS+rDiN5lX1eFBo6G8fKzXdRRz4kr5B9kO4pspM7b4ZOBg4c6A5RQ8IZJ1tuuWWhJ545cyYng2bHB\/LqriFf1rx6sCxVxiIvCo0sOUPg21r0rcAIVLcLyYtCQ4U+88w4y+Sf\/\/xn2gv69u3LyaCZ4zSbM+TLahaCouIWeVFoKAwcMatF34qIS0LyotDIsiU3btwoq1evdqXUqVNHKlasWGyJ7777rixZskR23XVXd\/Ba4uqUDRs2yLx589w9sQlY\/fr1i71fSCOHdL5iK14GMpCXzogWeVFo6GwcK7dF34rFCuWG5BWyD4rJxJedN0MnqNA999wjo0aNckfDn3DCCe4wNZx1ctttt8nOO++ckheu+dvf\/ubOSVm\/fr27FtuZw1Br1qxxJ7+uXbtWtt9+e+cIo0ePlrZt2xbJPqSRQzpfPjhMtutAXjrCFnlRaOhsHCu3Rd+KxYpCo2jyeSM0sCHXFVdc4QTGqlWrpHbt2m7rcQyZQGRgaCU5\/fLLL7LPPvvIpZdeKr1795YKFSpI+\/bt3Z8BAwa4P7NmzXLbmteoUUMGDx4sEyZMkIULF0qVKlXSkqHQiPe6snHTsbfIi0JDZ+NYuS36VixWFBpGhAZOaK1Xr57ceuutMnz4cPnqq6\/k9ttvl1dffVXOPfdcmThxojRr1qzQ0+BFuOmmm+TOO+90Z6H8\/vvv0qVLF7cfB+6BvTl69Oghffr0cdfNnz9fsIy2uJNgKTTiva5s3HTsLfKi0NDZOFZui74VixWFhhGhkSgKEoXGDz\/84OZdYEgFkYp06eGHH5ZFixa5g9ieeuopt0S2cePGMnLkSOnYsaO7bPny5W6vjnvvvVc6d+5cZEQj+Zc4th5iSJu+\/vprJ6CYMiNAXplx8rks8mrXrp08\/vjjUmfGDKk9Y0bBA6\/q0EFWduggvXr1Eqwyy0ayyCsbHDK5J1llQul\/eUrDCysuH3300UIFhtpiQfcU2cmdN0MnF110kRMK2E9jzJgxBRENCIwhQ4YUzLtIhwHDJ3PnznWTPgcNGiRHHHGEtG7d2i2ZPeyww9xlmGjaqFEjF+1A5CNdYkQjO86WyV35FZUJpf\/lsciLEQ2djWPltuhbsVih3JC8QvZBMZn4svNGaEC9+YgFJnZWrlzZrR6B8Y477jg3PJKc\/vjjD8GfxHNRrrzySjfM8tZbb8lBBx0kw4YNk27durlLfXRkypQp0qRJEwqNfPDApDqEfFnz8PGCV8kiLwqN4G6QlRta9K2sgMjwpiF5UWhkCL0k2b744gs3rIG5FAhD7b333m5yKEKpEB7JCYKhX79+8tFHH0nVqlXdr3EwG3YYRXSjw3\/DsPg\/Epa59uzZ0y2FLeqAtpBGDul8JWFq7Rry0lnMIi8KDZ2NY+W26FuxWDGiUTT5vIlolMRBsGwVczEwdwITRrFa5dprrxXsnYEx3qFDh8oDDzwgkydPlrp16wqGZ7Avx7hx44osjkKjJNYIcw0bNx1Hi7woNHQ2jpXbom\/FYkWhYUBoQDBg+eqHH34o3377rey+++6y7777SvPmzaVhw4ZFPsGDDz7o9tnwqUGDBi4qgomgWP6KZa+zZ892v8bvxo4d60RHUYlCI97rysZNx94iLwoNnY1j5bboW7FYUWjkudDAcMZ5553nNtpCwgoNCAS\/O+g111zjfl9UwsZcX375pTsLBXtuJO8mitUm2MwLQiOTRKGRCaXs5GHjpuNqkReFhs7GsXJb9K1YrCg08lhofP\/993LAAQe4CINfKeJFwieffOKWtGJi5\/XXXy9nn312znyIQiNnqDcriI2bjr1FXhQaOhvHym3Rt2KxotDIY6GB\/S4wpwI7d2KoIzn99ttvTmAgwpFqZ9BsORWFRrbIFn9fNm7FM0rMYZEXhYbOxrFyW\/StWKwoNPJYaGAXUGw9vmDBgrS1nDp1qttivLiVIiEdjEIjJE3dvdi4lX1eFBo6G8fKzXdRRz4kr5B9kO4pspM76qoTDJe8\/\/77RUYrMJETu3LifJKaNWtmh0LSXUMaOaTz5eThIxdCXjoDWORFoaGzcazcFn0rFitGNPI4okGhEfO1yM+y2bjp7GKRF4WGzsaxclv0rVisKDTyXGhgaOS0005LW0vsGIo8jGjEfIVyVzYbNx1ri7woNHQ2jpXbom\/FYkWhkedCA+eaZJIoNDKhZD8PGzedDS3yotDQ2ThWbou+FYsVhUYeCw2sJsHKkkxSjRo1MskWJA\/naATBWKKbsHHTYbPIi0JDZ+NYuS36VixWFBp5LDRiOkVRZVNoxLMMGzcde4u8KDR0No6V26JvxWJFoUGhofY9Cg01smAXsHHTobTIi0JDZ+NYuS36VixWFBoUGmrfo9BQIwt2ARs3HUqLvCg0dDaOlduib8ViRaFBoaH2PQoNNbJgF7Bx06G0yItCQ2fjWLkt+lYsVhQaRoTGHXfcITvuuKN06tRJatWqFdNfhEIjHn42bjr2FnlRaOhsHCu3Rd+KxYpCw4jQuPzyy+W5555ztW3fvr0ce+yx0rZtW6lWrVrOfYdCI+fICwpk46Zjb5EXhYbOxrFyW\/StWKwoNIwIDVRz2bJl8tJLL8nkyZPls88+czXv0aOHdOvWTZo3b54zH6LQyBnqzQpi46Zjb5EXhYbOxrFyW\/StWKwoNAwJjcSqYkdQHLg2YsQI9+OGDRvKueeeK8cdd5xUrVo1q\/5EoZFVvEXenI2bjr1FXhQaOhvHym3Rt2KxotAwJjTWrVsnOEht+vTpbutxpH333Vfq16\/vjpNv166dPPjgg1n1JwqNrOKl0AiI12JnQKER0AGyeCuLvpVFHMXeOiSvkH1QsRXPQYaop7cmPh\/Exfjx4+WFF15wP65Xr56cdNJJcswxx0iDBg3cz9555x058cQTBdGObKaQRg7pfNl85ny5N3npLGGRF4WGzsaxclv0rVisGNEwEtHAUfCLFi1y8zG6du0qzZo1kwoVKhSq\/YoVK+T22293f7KZKDSySbfoe7Nx07G3yItCQ2fjWLkt+lYsVhQaRoTGm2++KbvvvrvUqVNnsxp\/9dVXstNOO0nFihVz4kcUGjnBnLIQNm469hZ5UWjobBwrt0XfisWKQsOI0EBE4\/DDD5ezzz67UI3h7JiXgdUoe+yxR078iEIjJ5gpNAJgttgZUGgEMHwObmHRt3KAJW0RIXmF7INiMvFlR5+jgVUkWMqKSaBIW221VSEu\/udz586VunXr5oRZSCOHdL6cPHzkQshLZwCLvCg0dDaOlduib8VixYhGnkc0Ro0aJWvXrpVnn31WdtllF2nZsmWhGmMp6yGHHCItWrTImQ9RaOQM9WYFsXHTsbfIi0JDZ+NYuS36VixWFBp5LjR89caOHSt77rnnZkIjhuNQaMSg\/p8y2bjp2FvkRaGhs3Gs3BZ9Kxar0G1XyD4oJhNfdtShk4ULF8qnn37qVppgaOTbb79Ny+T444+XKlWq5IRZSCPzZdWZjLzKPi8KDZ2NY+Xmu6gjH5JXyD5I9xTZyR1VaAwaNEjGjBkjS5culYsvvlhmzJiR9ikhSmrWrJkdCkl3DWnkkM6Xk4ePXAh56QxgkReFhs7GsXJb9K1YrBjRKJp8VKGxYcMG+e2339wEUPx748aNaWtbvXr1nPkQhUbOUG9WEBs3HXuLvCg0dDaOlduib8ViRaGRx0IDG3CtWbMmI9\/AHhtbbLFFRnlLm4lCo7QES349GzcdO4u8KDR0No6V26JvxWJFoZHHQsMPnWTiHBw6yYSS\/Txs3HQ2tMiLQkNn41i5LfpWLFYUGnksNODI\/\/rXvzLyDSxvrVSpUkZ5S5uJEY3SEiz59WzcdOws8qLQ0Nk4Vm6LvhWLFYVGHguNmE5RVNkUGvEsw8ZNx94iLwoNnY1j5bboW7FYUWjksdDA4WiPPvqoYFjk2muvdduMp0tvvPGGbL311jnxIwqNnGBOWQgbNx17i7woNHQ2jpXbom\/FYkWhkcdC45VXXpHFixdLv379ZOrUqW5PjXSpb9++Uq1atZz4EYVGTjBTaATAbLEzoNAIYPgc3MKib+UAS9oiQvIK2QfFZOLLjrq8NRUArEL55ptv5Pfff5f69evLNttsk3NOIY0c0vlyDiJCgeSlg26RF4WGzsaxclv0rVisGNHI44hGYtV++eUXufnmm+XJJ58sVOPOnTvL9ddfn7MD1VA4hUa815WNm469RV4UGjobx8pt0bdisaLQMCI0hg4dKg888IB06dLFHRePDboWLVrkDlvbYYcd5LnnnuOqk5hvUY7KZuOmA22RF4WGzsaxclv0rVisKDSMCI1DDz3Und761FNPFarxBx98IF27dpVp06ZJ48aNc+JHjGjkBHPKQti46dhb5EWhobNxrNwWfSsWKwoNI0KjXbt2cuSRR0r\/\/v0L1fj777+XAw44wEU2mjdvnhM\/otDICWYKjQCYLXYGFBoBDJ+DW1j0rRxgSVtESF4h+6CYTHzZeTMZ9J577pFRo0a5g9Xq1q3r6odzUIYNGyaTJk2SN998k0Mn+eAxWa5DyJc1y1XNi9tb5EWhkReuU2wlLPpWsQ+VxQwheVFoBDQUhMWsWbPcHbHKZMGCBe7fLVu2lAoVKsiSJUtk3bp10qBBA5kyZYo7fC0XKaSRQzpfLp49dhnkpbOARV4UGjobx8pt0bdisUK5IXmF7INiMsmLiMYzzzwj8+bNy4jDTTfdRKGRESnbmUK+rLZJZFZ7i7woNDKzbexcFn0rJrOQvCg0IlgSkY4mTZqk3bDrs88+k3feeTL7QmgAACAASURBVEfWr18v++23nzRt2rSgljh+HmJm9erVLlKCvTmKSyGNHNL5iqt3Wfg9eemsaJEXhYbOxrFyW\/StWKwY0SiafN7M0UA13333Xfn888\/ljz\/+KKj1r7\/+6vbRwNAJxEZywo6il156qRtewWZfEBTYj6Nnz57u\/506dZK1a9fK9ttv70Jbo0ePlrZt2xZJhUIj3uvKxk3H3iIvCg2djWPltuhbsVhRaBgRGli+evHFF6es7V577SWTJ0+WypUrb\/b7Nm3aOOEwcOBAN3kUq1befvttN\/cDP8PfuHeNGjVk8ODBMmHCBHe2SpUqVdKSodCI97qycdOxt8iLQkNn41i5LfpWLFYUGkaExuWXXy5YynrLLbfI6aefLmeccYYcdthhgrkZu+22m\/s7Ofmlr+PHj3fDIkgTJ06UK6+80g2XdOvWTXr06CF9+vRxv5s\/f750795dEvOnwkOhEe91ZeOmY2+RF4WGzsaxclv0rVisKDSMCA1s2IUhkFNOOUWuu+46F71AROKLL75wEQtEKWrVqlXk0\/z888\/Sq1cvd8orVrRgg6+RI0dKx44d3XXLly+XVq1ayb333ivY2jxdQkOYnCB8IIC06euvv5Z69eppLyu3+clLZ3qLvLBnzuOPPy51ZsyQ2jNmFDzwqg4dZGWHDu4dnjlzpg5Ehrkt8srw0YJnIysd0tLweuyxx9xJ5okJ0wjKSsqbORroyHfccUcZMmSIEwmYk\/H8888LzkDZZ599BIZAhCNdmjNnjjtqHvkxD6N27drSunXrQtdt3LhRGjVqJMOHD3dbnRclNEIZmV8FuleFvMo+L0Y0dDaOlZvvoo58SF4ho+q6p8hO7rwRGlBzGB658MIL5cQTTxR89ZxwwgkCcYAJn5hXUbNmzc0oYF4GJn9CiCDicNlll7kTXzERtFmzZm7DLwyhIP3www\/SokWLtBNL\/c1DGjmk82XHBfLrruSls4dFXhQaOhvHym3Rt2KxQrkheYXsg2Iy8WXnjdCAYHj44YddRAJDKHfddZc89NBDrp6XXHKJnHfeeSl5Ie\/cuXNdFANLWxMTlrkiDOu3Nce8DaxGwUZg1apVY0QjHzwwqQ4hX9Y8fLzgVbLIi0IjuBtk5YYWfSsrIDK8aUheFBoZQg+RDaIDKZ0o8JNBL7jggs2WrCKaceedd7oTYbFiBduaX3TRRVKxYkUZN25ckdULaeSQzheCab7fg7x0FrLIi0JDZ+NYuS36VixWjGgUTT5vIhqo5pdffulOb8UEUGxJjgbpuOOOc\/MqUqXXXntNzjnnnJS\/e\/3112W77baT3r17y+zZs10e7LUxduzYgrNU0qGh0Ij3urJx07G3yItCQ2fjWLkt+lYsVhQaRoQGdv\/EihMkRB9wrgl2\/ETC6pOSrPjwj47VJtg1FEIjk0ShkQml7ORh46bjapEXhYbOxrFyW\/StWKwoNIwIDexvgbkTzz33nDRs2NDV+qeffpIRI0a4+RdvvfWW1KlTJyd+RKGRE8wpC2HjpmNvkReFhs7GsXJb9K1YrCg0jAiN5ImbvtrYPhyTPJ944gnBXhu5SBQauaCcugw2bjr2FnlRaOhsHCu3Rd+KxYpCw4jQwD4aSMmblixevFiOP\/54RjRivkE5LJuNmw62RV4UGjobx8pt0bdisaLQyGOhgU2xVq5c6Wq4dOlSt48GJnceccQRssUWW8inn34qQ4cOdRt2YRInVozkIjGikQvKjGiEoGyxM6DQCGH57N\/Dom9ln0r6EkLyCtkHxWTiy4666gQ7eWKVSSYp3YZdmVyrzRPSyCGdT\/scFvOTl85qFnlRaOhsHCu3Rd+KxYoRjTyOaGAfDOzgmUnaZZddXJQjF4lCIxeUGdEIQdliZ0ChEcLy2b+HRd\/KPhVGNErCOGpEI7nC2CIcEY5ly5a5fTSwHPXYY4\/NeFlqSQCkuoZCIxRJ\/X3YuOmYWeRFoaGzcazcFn0rFitGNPI4opFYNWzShYPO1q1b5zbaQlq9erX7O\/G8klw4EoVGLigzohGCssXOgEIjhOWzfw+LvpV9KoxolIRx3kQ0sD34Rx995M432W233dyzrFixQu677z63tJVzNEpiXnvXsHHT2cwiLwoNnY1j5bboW7FYMaJhJKLRsmVL6devn5x66qmFauyPicf5JAcffHBO\/IgRjZxgTlkIGzcde4u8KDR0No6V26JvxWJFoWFEaLRp00awl0by2SX+uHfsDtq2bduc+BGFRk4wU2gEwGyxM6DQCGD4HNzCom\/lAEvaIkLyCtkHxWTiy86boZPBgwfLhAkT3F4aBx10kNtu\/P3335dRo0bJjBkzOHSSD96SgzqEfFlzUN3oRVjkRaER3W0yqoBF38rowbKUKSQvCo0sGQlDJIhmzJs3b7MShg8f7iaK5iqFNHJI58vV88csh7x09C3yotDQ2ThWbou+FYsVh06MDJ34akJofPjhh4KlrvXq1RMMqey444459R8KjZziLlQYGzcde4u8KDR0No6V26JvxWJFoWFEaJx00kmy7777yg033BDTV1zZFBrxTMDGTcfeIi8KDZ2NY+W26FuxWFFoGBEafo7Gm2++KdWrV4\/pLxQaEemzcdPBt8iLQkNn41i5LfpWLFYUGkaExvTp06Vv375uF1Cc1lqzZs1CNe\/WrZtUrVo1J37EiEZOMKcshI2bjr1FXhQaOhvHym3Rt2KxotAwIjSwtHX27Nlpa8sNu2K+Qrkrm42bjrVFXhQaOhvHym3Rt2KxotAwIjRwtsmmTZvS1rZSpUo58yFGNHKGerOC2Ljp2FvkRaGhs3Gs3BZ9KxYrCo08FxobNmyQmTNnyvLly+XQQw+Vxo0bx\/QVVzaFRjwTsHHTsbfIi0JDZ+NYuS36VixWFBp5LjQwH2Px4sUFtdxvv\/0E241Xq1Ytms9QaERDL2zcdOwt8qLQ0Nk4Vm6LvhWLFYVGHguNDz74QLp27SpXX321tG7dWqZOnSojR46UQYMGyWmnnRbNZyg0oqGn0FCit9gZUGgojRwpu0XfioTKFRuSV8g+KCYTX3bULcjvuusuGTt2rCxYsMDVB3M0DjzwQDn66KNl4MCB0fiENHJI54sGJIcFk5cOtkVeFBo6G8fKbdG3YrGi0MjjiAYiF++8845MmjSpoJZnnnmm1K1bV4YMGRLNZyg0oqEP+lUQ7ylyV7LFzoBCI3f+UZqSLPpWaZ63tNeG5BWyDyrtc4W4PmpEA0IDB6c9\/fTTBc+C805woBqFRgjz2rtHyJfV3tPra2yRF4WG3s4xrrDoWzE4+TJD8qLQCGhJCI1FixYJjoD36fLLL5dtttlGbrzxxkIl1a5dO2DJRd8qpJFDOl\/OAEQsiLx08C3yotDQ2ThWbou+FYsVyg3JK2QfFJOJLzt6RGPMmDEZceCGXRlhMp8p5MtqHkYGD2CRF4VGBobNgywWfSsmtpC8KDQCWnLOnDmClSeZpLPOOotbkGcCyniekC+rcRQZVd8iLwqNjEwbPZNF34oJLSQvCo2YlsxR2SGNHNL5cvT4UYshLx1+i7woNHQ2jpXbom\/FYsWhk6LJRx06iekURZVNoRHPMmzcdOwt8qLQ0Nk4Vm6LvhWLFYUGhYba9yg01MiCXcDGTYfSIi8KDZ2NY+W26FuxWFFoUGiofY9CQ40s2AVs3HQoLfKi0NDZOFZui74VixWFBoWG2vcoNNTIgl3Axk2H0iIvCg2djWPltuhbsVhRaFBoqH2PQkONLNgFbNx0KC3yotDQ2ThWbou+FYsVhQaFhtr3KDTUyIJdwMZNh9IiLwoNnY1j5bboW7FYUWhQaKh9j0JDjSzYBWzcdCgt8qLQ0Nk4Vm6LvhWLFYUGhYba9yg01MiCXcDGTYfSIi8KDZ2NY+W26FuxWFFoUGiofY9CQ40s2AVs3HQoLfKi0NDZOFZui74VixWFBoWG2vcoNNTIgl3Axk2H0iIvCg2djWPltuhbsVhRaFBoqH2PQkONLNgFbNx0KC3yotDQ2ThWbou+FYsVhQaFhtr3KDTUyIJdwMZNh9IiLwoNnY1j5bboW7FYUWiUE6Gxdu1amTp1qpx88slSqVKlgqfesGGDzJs3T1avXi0tW7aU+vXrF+uLFBrFIspaBjZuOrQWeVFo6GwcK7dF34rFikKjnAiNO+64Q0aMGCFLliyRatWquades2aNdOrUSSBCtt9+e8GLM3r0aGnbtm2RVCg04r2ubNx07C3yotDQ2ThWbou+FYsVhUYZFxoTJ06U22+\/XVasWOGeNFFoDBgwQGbNmiXTpk2TGjVqyODBg2XChAmycOFCqVKlSloyFBrxXlc2bjr2FnlRaOhsHCu3Rd+KxYpCo4wLjQ8++ECWLVvmxMO4ceMKCY02bdpIjx49pE+fPo7C\/PnzpXv37jJ+\/Hg3jJIuUWjEe13ZuOnYW+RFoaGzcazcFn0rFisKjTIuNPzjTZ8+Xfr27VsgNDA3o3HjxjJy5Ejp2LGjy7Z8+XJp1aqV3HvvvdK5c2cKjZhvZZqy2bjpjGKRF4WGzsaxclv0rVisKDTKqdD45ptvpHXr1vLYY4\/JYYcd5ihs3LhRGjVqJMOHD5cuXboUKTSSf3nGGWfI6aefrvbjr7\/+WurVq6e+rrxeQF46y1vk1a5dO3n88celzowZUnvGjIIHXtWhg6zs0EF69eolM2fO1IHIMLdFXhk+WvBsZKVDWhpe6KceffTRQgV+\/vnnugrkce4KmzZt2pTH9cu4askRDUwEbdasmQwbNky6devm7vPDDz9IixYtZMqUKdKkSRNGNDKmm7uM\/IrSsbbIixENnY1j5bboW7FYMaJRTiMaeOymTZu6r6P+\/fs7Cljm2rNnz0LzOFLh4RyNeK8rGzcde4u8KDR0No6V26JvxWJFoVGOhcbQoUPlgQcekMmTJ0vdunXloosukooVK7pJo0UlCo14rysbNx17i7woNHQ2jpXbom\/FYkWhUY6Fxi+\/\/CK9e\/eW2bNnOwoNGjSQsWPHOtFBoRHzlUxfNhs3nV0s8qLQ0Nk4Vm6LvhWLFYVGOREaRT0mVpusX7\/eCY1MEiMamVDKTh42bjquFnlRaOhsHCu3Rd+KxYpCg0JD7XsUGmpkwS5g46ZDaZEXhYbOxrFyW\/StWKwoNCg01L5HoaFGFuwCNm46lBZ5UWjobBwrt0XfisWKQoNCQ+17FBpqZMEuYOOmQ2mRF4WGzsaxclv0rVisKDQoNNS+R6GhRhbsAjZuOpQWeVFo6GwcK7dF34rFikKDQkPtexQaamTBLmDjpkNpkReFhs7GsXJb9K1YrCg0KDTUvkehoUYW7AI2bjqUFnlRaOhsHCu3Rd+KxYpCg0JD7XsUGmpkwS5g46ZDaZEXhYbOxrFyW\/StWKwoNCg01L5HoaFGFuwCNm46lBZ5UWjobBwrt0XfisWKQoNCQ+17FBpqZMEuYOOmQ2mRF4WGzsaxclv0rVisKDQoNNS+R6GhRhbsAjZuOpQWeVFo6GwcK7dF34rFikKDQkPtexQaamTBLmDjpkNpkReFhs7GsXJb9K1YrCg0KDTUvkehoUYW7AI2bjqUFnlRaOhsHCu3Rd+KxYpCg0JD7XsUGmpkwS5g46ZDaZEXhYbOxrFyW\/StWKwoNCg01L5HoaFGFuwCNm46lBZ5UWjobBwrt0XfisWKQoNCQ+17FBpqZMEuYOOmQ2mRF4WGzsaxclv0rVisKDQoNNS+R6GhRhbsAjZuOpQWeVFo6GwcK7dF34rFikKDQkPtexQaamTBLmDjpkNpkReFhs7GsXJb9K1YrCg0KDTUvkehoUYW7AI2bjqUFnlRaOhsHCu3Rd+KxYpCg0JD7XsUGmpkwS5g46ZDaZEXhYbOxrFyW\/StWKwoNCg01L5HoaFGFuwCNm46lBZ5UWjobBwrt0XfisWKQoNCQ+17FBpqZMEuYOOmQ2mRF4WGzsaxclv0rVisKDQoNNS+R6GhRhbsAjZuOpQWeVFo6GwcK7dF34rFikKDQkPtexQaamTBLmDjpkNpkReFhs7GsXJb9K1YrCg0KDTUvkehoUYW7AI2bjqUFnlRaOhsHCu3Rd+KxYpCg0JD7XsUGmpkwS5g46ZDaZEXhYbOxrFyW\/StWKwoNCg01L5HoaFGFuwCNm46lBZ5UWjobBwrt0XfisWKQoNCQ+17FBpqZMEuYOOmQ2mRF4WGzsaxclv0rVisKDQoNNS+R6GhRhbsAjZuOpQWeVFo6GwcK7dF34rFikKDQkPtexQaamTBLmDjpkNpkReFhs7GsXJb9K1YrCg0KDTUvkehoUYW7AI2bjqUFnlRaOhsHCu3Rd+KxYpCg0JD7XsUGmpkwS5g46ZDaZEXhYbOxrFyW\/StWKwoNCg01L5HoaFGFuwCNm46lBZ5UWjobBwrt0XfisWKQoNCQ+17FBpqZMEuYOOmQ2mRF4WGzsaxclv0rVisKDQoNNS+R6GhRhbsAjZuOpQWeVFo6GwcK7dF34rFikKDQkPtexQaamTBLmDjpkNpkReFhs7GsXJb9K1YrCg0KDTUvkehoUYW7AI2bjqUFnlRaOhsHCu3Rd+KxYpCg0JD7XsUGmpkwS5g46ZDaZEXhYbOxrFyW\/StWKwoNCg01L5HoaFGFuwCNm46lBZ5UWjobBwrt0XfisWKQoNCQ+17FBpqZMEuYOOmQ2mRF4WGzsaxclv0rVisKDQoNNS+R6GhRhbsAjZuOpQWeVFo6GwcK7dF34rFikKDQkPtexQaamTBLmDjpkNpkReFhs7GsXJb9K1YrCg0yrnQ2LBhg8ybN09Wr14tLVu2lPr16xfrixQaxSLKWgY2bjq0FnlRaOhsHCu3Rd+KxYpCoxwLjTVr1kinTp1k7dq1sv322wtenNGjR0vbtm2LpBJSaNx0001y4403xvR\/U2WTl85cFnnFFBoWeek8IlxustKxDMkrZB+ke4rs5K6wadOmTdm5dfy7DhgwQGbNmiXTpk2TGjVqyODBg2XChAmycOFCqVKlStoKhjRyyHvFJ5r9GpCXjrFFXjGFhkVeOo8Il5usdCxD8gp5L91TZCd3mRYabdq0kR49ekifPn0cvfnz50v37t1l\/PjxbhglXQpp5JD3yo4L5NddyUtnD4u8KDR0No6V26JvxWKFckPyCnmvmEx82WVWaGBuRuPGjWXkyJHSsWNH97zLly+XVq1ayb333iudO3em0MgHD0yqQ1l7wbKN2CIvCo1se0WY+1v0rTBPXrK7hOQV8l4le5qwV5VZofHNN99I69at5bHHHpPDDjvMUdu4caM0atRIhg8fLl26dElLsmfPnm4CKRMJkED2CDT69ddCN\/+4iOHM7NWCdyaB\/CNw8MEHy7hx4\/KvYiWsUZkVGpgI2qxZMxk2bJh069bN4fnhhx+kRYsWMmXKFGnSpEkJkfEyEiABEiABEiCBTAmUWaEBAE2bNpVevXpJ\/\/79HQ9EKRCtWLJkiVSrVi1TRsxHAiRAAiRAAiRQQgJlWmgMHTpUHnjgAZk8ebLUrVtXLrroIqlYsWKZCkmV0O68jARIgARIgARyQqBMC41ffvlFevfuLbNnz3YwGzRoIGPHjnWig4kESIAESIAESCD7BMq00PD4sNpk\/fr1TmgwkQAJkAAJkAAJ5I5AuRAaucPJkkiABEiABEiABBIJUGjQH0iABEiABEiABLJGgEIja2h5YxIgARIgARIgAQoN+gAJkAAJkAAJkEDWCFBoZA0tb0wCJEACJEACJEChkUUfwE6kc+bMkapVq7pD3GrWrJnF0uzc+o8\/\/pD3339fli5dKvvss4\/7U6FChZQPgCXKOAzvs88+c8uScVBe9erV7TxsKWuqYeWLwoHMWMaNXXBx3k95Slpen3\/+uXtHd9ppJzniiCPcPjvlKWl5ffDBB\/Lee+\/JIYccwlV8SY6ydu1amTp1qpx88slSqVKl8uRGxT4rhUaxiEqWYfHixXL88cdLvXr1BJ0l0rPPPiu77LJLyW5Yhq46\/\/zz5ZVXXpG9995bPvzwQ7fXid+9NfExsY38GWecIWDp80JszJw5s9zs7Jopq0RuEBk33HCD3HbbbXLKKaeUIc8p\/lE0vO666y65++673UfAggULXMf5wgsvuA+D8pIy5fXbb79J165d3ccBOP3jH\/+Q4447Tu68887ygqrY57zjjjtkxIgR3Hk6BSkKjWLdR58BXwkdOnRwnSMaMrykEB34aho1apT+hmXoCjTk2KF12rRp7mv75ZdfdkIj1fkz+Dq49NJLZdasWU6gffLJJ9KpUyfXuKGRK+tJw8qz8Izw\/\/ImNDS8PvroIzn66KNl9OjR0rZt24KTnRNPe6Z\/\/Y\/Aiy++KBdeeKHMnTvXRRanT58uffv2lRkzZkjDhg3LOqoin2\/ixIly++23y4oVK1w+HnGxOS4KjSy8Iv7k2KefftqFr5FwND06SHwRVCnHp1Red9118tVXX7lTdZE2bNjgBMeVV17pGrLEdOutt8rHH38sjz76aMGP8fXZuXNnuemmm7Jgufy6pYYVao7IGb46IWofeeQRx7Q8RTQ0vO6\/\/36577775N1335V169bJ1ltvLStXrnTRDPy7PCQNryeeeEIGDBgg77zzjmyzzTZOcJx22mkCAbLnnnuWB1xpnxHDScuWLZOFCxe64y0oNCg0cvJC+MPb3n77balVq5Yr03+d+y+CnFQkDwtBx4eTcxHa9wnzLvBVOXDgwCJr7L+iEgVcHj5isCppWUF8YSgKQycYQy9vQkPD6+qrr3bDJRAWfjige\/fucs4558gWW2wRzIb5fCMNL8w3O\/bYY93j7Lvvvm6IqV27dvLggw\/m8yPmtG6+faLQoNDIieM999xzcvnllxdStl584NyVnXfeOSf1yMdCDj30UDnxxBPliiuuKKgeGniEX2+55ZaUVcZcjeHDh7uvdORFpKM8JA2rV1991Q0zobHDvCBEfsqb0NDwOumkk9wX6AUXXCBHHnmkvPTSS+4AxmHDhkm3bt3Kg3uJhhfmSZ177rmOS7Nmzdw8KbyzmHdWXiJAxTkFhUZ6Qhw6Kc57SvB7NPp4KfEy+vNVMPkRE68+\/fTTcvPFlArdMcccI3vttZcb0\/QJQyH4usLEz+SEsORZZ53lJn8OHjxYDj\/88BJYxOYlGlZNmzYV\/MEcFiREh8AV9\/A\/s0kh81preJ1++umCeRr4APArTfCF3rx580K+mXnp9nJqeGFe1aJFi+T11193Kyr88DBOyIZoY5KCeSuMaDCikZP3AUvm2rdv777AfceIL3IMm4wfPz4ndcjXQhDpAR9EfZB+\/vlnF4rFGDC+sBITlsAiXHvmmWfKVVddVa5WA4CDhhW+zLGs1ScIW3xxovPEMEF5SBpeYIL3ER2nT5jAjUhQeYmYaXjBjzDkiXbMJ7yvHTt2LHbIszz4Hp6REY30lmZEI0tvAb4WoPwxi33VqlXuqxxf7PgyKM\/JR3sQokYjdc8998hTTz3l9jLAJDMIkO+\/\/17OPvtsF8GYMGGCGwdO3N+gTp06sttuu5V5jBpWyTDK49CJhpcfyrzmmmvcCiasnrj++uvdqjB8JJSHpOEFYYZ38aGHHpIDDjjA8cLQHIRHly5dygOuYp+RQoNCo1gnCZ3hyy+\/dPMJ\/JInfC1hjTU3chHx683BfKuttnJj4z6acfHFF7s1+s8\/\/7wL+2NyY3LChjhDhgwJbbK8vF+mrJIrD56Ys1GeVp2AgYYXVp0kDuHB9y677LK89INsVSpTXpgMCuHvI5GoD4afsBKlvEyeLc4GFBoUGsX5SFZ+j\/000GlishS+wpn+RwC76H399deyxx57UHwV4xhkpXtzNLww0RjzDTCBtkaNGrqCykhuDS8sR8dSzh122IFtWhmxfy4eg0MnuaDMMkiABEiABEignBKg0CinhudjkwAJkAAJkEAuCFBo5IIyyyCBPCCApYgYrkqXMDHXp59++slNjsQkQEyQxNDCeeedl\/JSnBmCQ+8SU+XKld2Or1haW79+\/RI9PTbUwg6y2Eq9PB2kVyJYvIgE8pgAhUYeG4dVI4GQBLCvBrZ\/x2m5qRJW\/\/iEnWyvvfZat0V369at3R4dmLSbKmEfBeyumbg8+ccff3Q7b2Ky7zPPPCONGjUq9lGWL18urVq1Klj5gd0nsawZS1B58nGx+JiBBPKWAIVG3pqGFSOBsAQgNLArbSbbRmMZNrbPHzRokBMQxQkNHBw4adKkQhX2+6Dg8K3EnWCRCXt+VKhQoVB+vwlUcQebYZI1Elc7hPUP3o0EskWAQiNbZHlfEsgzApkKDRzOhqgHDrNDNKOkQsPfJ3E5Mu6JpcvY\/nu\/\/faT\/\/u\/\/3OH6X333Xdunxms0sLpoNhZFyuScBAhrsHKLSwVx3JKbEaGhLrdeOON8uc\/\/znPSLM6JEACiQQoNOgPJFBOCEBoYAlnqsPrateuXbBc0Z+bgkMBcehYSYQGRAZ2e8X8CuzVgBNlX375Zendu7f06NFDjjjiCMH9Mf8DO3Fi0ycMseBgOGxuh\/zYrh+7VyIf5mhgIy1ETjB35PfffxfMDcHyVGz1z\/M2yokT8zFNEqDQMGk2VpoE9AQgNDCXIlXC5l74g4QJoL\/++quLJiBlIjQQocCEUZ\/8pNOjjz7aCQLs7IrNnjCc4k\/u3bhxo+y\/\/\/7uuHHMxUgeOvGHE0JoYJ4GREfiVvU4qwT3x0ZSp556qh4IryABEsgJAQqNnGBmISQQnwCEBr780TEnJ0y2RFQDEQMcLPbXv\/614DC2TIQGRAIiET6tXLlSnnzySVm3bp0b6vDDG9j6G6IBO+e+9957bhUMzmkpTmhgXgkmoyYeWIWoBoZXcBYOhlSYSIAE8pMAhUZ+2oW1IoHgBDKZo+HPAIEI8EtKMxEaqSaDTpw40Z2HgQmliFo8\/PDDcvPNN0uLFi3k4IMPlj333NOtbMkkooHhlylTpjhx4ieRgj7H0AAAAa5JREFUIuqCk4DPOeccue6664Lz4g1JgATCEKDQCMORdyGBvCeQidBAtANLYDF3wqeSCg1\/Mq8flsFBbzgM7+mnn3a39pNFM4loYP4Gzrd58cUXnUBB8qtaIF569uyZ9\/xZQRIorwQoNMqr5fnc5Y4AhAbmRWBFR6p07LHHupUciDJgMmZphUby0AYOeMPJvDgBFCIDJ39COODwQUQ9sPfGgQce6IZScHrvG2+8UTAZFEMwhx9+uBvOwTAJ7o16Llq0qODk33JnUD4wCRghQKFhxFCsJgmUlkBRk0FxbywjxRJTTL7EHhoaoYG8PlKRWE\/sv4E0f\/58t6QVp6OuXr3a\/QzLXnGa8bhx49zqFAgRnAg6Z84cJzaws2i\/fv0K6jNt2jTBCas+YTMwzN3AMAwTCZBA\/hKg0Mhf27BmJFDmCCAS8cknn7htyf0ckG+\/\/Va22247t5QWG3mtWrXKTVrF\/5MTIiHY7rxatWqy66678uTfMuchfKCySIBCoyxalc9EAiRAAiRAAnlCgEIjTwzBapAACZAACZBAWSRAoVEWrcpnIgESIAESIIE8IfD\/zlGnqZoHpnwAAAAASUVORK5CYII=","height":324,"width":538}}
%---
%[output:7eca2388]
%   data: {"dataType":"text","outputData":{"text":"Assigning cortical layers based on channel depth...\nCalculating auto- and cross-correlograms for 26 neurons...\nCorrelation analysis complete. Generating visualizations...\n","truncated":false}}
%---
%[output:4f8ca4a3]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAhoAAADgCAYAAABFL89wAAAAAXNSR0IArs4c6QAAF8NJREFUeF7t3X9sVWf9wPEPhaGNZabJMPxhGG0M2TIXQnRJYwZG0IW5GKpMt+FGzJJthDG7seHmHFDG3JgFZRuOMIwJzWY6jbWxgMZJNMU\/iD9GpEAjySpRCJa4LWMzxNLCN+fkS0O3ymXX+0R67usmja6e+3jP63wefef+6J1w9uzZs+FGgAABAgQIEEggMEFoJFC1JAECBAgQIJALCA2DQIAAAQIECCQTEBrJaC1MgAABAgQICA0zQIAAAQIECCQTEBrJaC1MgAABAgQICA0zQIAAAQIECCQTEBrJaC1MgAABAgQICA0zQIAAAQIECCQTEBrJaC1MgAABAgQICA0zQIAAAQIECCQTEBrJaC1MgAABAgQICA0zQIAAAQIECCQTqOrQGBoaikmTJiXDtTABAgQIEKh2gaoNje7u7ti6dWvs2LGj2mfA+RMgQIAAgWQCVRcax48fj7a2tnjllVfiyiuvFBrJRsvCBAgQIECgCr+99Y033oienp549dVX85+xntFYvHhx7N27N1paWvIfNwIECBAgQKA8gap7RuMc065du+L5558fMzQaGxujv7+\/PFH3IkCAAAECBEYEhMYY79EQGnYIAQIECBCojIDQEBqVmSSrECBAgACBMQSEhtCwMQgQIECAQDKBqg2NnTt3xpYtW7xHI9loWZgAAQIECFThp04u5qJ7j8bFKDmGAAECBAiUFqjaZzQuRCM0Sg+OIwgQIECAwMUICI0xlITGxYyOYwgQIECAQGkBoSE0Sk+JIwgQIECAQJkCQkNolDk67kaAAAECBEoLCA2hUXpKHEGAAAECBMoUEBpCo8zRcTcCBAgQIFBaQGgIjdJT4ggCBAgQIFCmgNAQGmWOjrsRIECAAIHSAkJDaJSeEkcQIECAAIEyBYSG0ChzdNyNAAECBAiUFhAaQqP0lDiCAAECBAiUKSA0hEaZo+NuBAgQIECgtIDQEBqlp8QRBAgQIECgTAGhITTKHB13I0CAAAECpQWEhtAoPSWOIECAAAECZQoIDaFR5ui4GwECBAgQKC0gNIRG6SlxBAECBAgQKFNAaAiNMkfH3QgQIECAQGkBoSE0Sk+JIwgQIECAQJkCQkNolDk67kaAAAECBEoLCA2hUXpKHEGAAAECBMoUEBpCo8zRcTcCBAgQIFBaQGgIjdJT4ggCBAgQIFCmgNAQGmWOjrsRIECAAIHSAkJDaJSeEkcQIECAAIEyBYSG0ChzdNyNAAECBAiUFihkaAwPD8fEiRMvePanT5+Oyy67bMxjGhsbo7+\/v7SeIwgQIECAAIELChQqNPbt2xerVq2KQ4cOxZw5c2LTpk1RX18\/CuCdd96Jp556Kn7+85\/n\/9nXvva1uPPOO0cdIzTsGgIECBAgUBmBwoTGqVOnYv78+bF06dJobm6O1tbWmDx5cqxfv36UVEdHR7z88suxdevW+Otf\/xq33XZb7N69OxoaGkaOExqVGS6rECBAgACBwoRGb29vLFy4MA4ePBi1tbWxZ8+ePDayiDj\/tmHDhjh8+HC88MIL8eabb8YnPvGJ\/NmNj3\/840LDfiBAgAABAhUWKExodHd3R1tbW\/T09ORER44ciXnz5sX+\/fujrq5uhO3YsWOxYMGC\/GWTo0ePRlNTU7z00ksxYcKEUaGR\/UNLS0v+40aAAAECBAiUJ1CY0Ojs7Iz29vbo6urKJU6cOJFHxLtD4zvf+U7s2LEjvv71r8c\/\/vGP+O53v+sZjfJmx70IECBAgEBJgcKERvZyyKJFiyJ7CSW7HThwIJYtWzbyDMc5iblz58a9994bt9xyS\/6ru+66K6655pq4\/\/77Rz2j4VMnJWfHAQQIECBAoKRAYUJjaGgoZs6cGdu3b4\/Zs2fH6tWro6amJrL3ZPT19cXg4GDMmjUrHn300Thz5kysXbs2so+43nDDDfHtb387PvOZzwiNkuPiAAIECBAg8P4EChMa2WlnL5usWLEiF8g+RZJ9wmTq1Kmxbt26GBgYiM2bN+d\/HyM75rXXXsuPu+mmm+KJJ56ISZMmCY33NzuOJkCAAAECJQUKFRrZ2WYfc82iYsaMGRc8+ePHj8eUKVNGvVH03B18vLXk3DiAAAECBAhclEDhQuOizrrEQUKjEorWIECAAAECEUJjjCkQGrYGAQIECBCojIDQEBqVmSSrECBAgACBMQSEhtCwMQgQIECAQDIBoSE0kg2XhQkQIECAgNAQGnYBAQIECBBIJiA0hEay4bIwAQIECBAQGkLDLiBAgAABAskEhIbQSDZcFiZAgAABAkJDaNgFBAgQIEAgmYDQEBrJhsvCBAgQIEBAaAgNu4AAAQIECCQTEBpCI9lwWZgAAQIECAgNoWEXECBAgACBZAJCQ2gkGy4LEyBAgAABoSE07AICBAgQIJBMQGgIjWTDZWECBAgQICA0hIZdQIAAAQIEkgkIDaGRbLgsTIAAAQIEhIbQsAsIECBAgEAyAaEhNJINl4UJECBAgIDQEBp2AQECBAgQSCYgNIRGsuGyMAECBAgQEBpCwy4gQIAAAQLJBISG0Eg2XBYmQIAAAQJCQ2jYBQQIECBAIJmA0BAayYbLwgQIECBAoJChMTw8HBMnTrzg1c2OGRwcjNra2vcc19jYGP39\/aaDAAECBAgQ+C8FChUa+\/bti1WrVsWhQ4dizpw5sWnTpqivr38P0QsvvBDbtm2L119\/PZYsWRKPPfZYTJo0aeQ4ofFfTpW7EyBAgACB\/xcoTGicOnUq5s+fH0uXLo3m5uZobW2NyZMnx\/r160dd7L1798Z9990X3\/ve92L69Olx++23x7p16+LTn\/600LAtCBAgQIBAhQUKExq9vb2xcOHCOHjwYP5yyJ49e\/LY2L179yiy1atX5\/\/8wAMPRPbyydmzZ\/Mg+fCHPzwqNLJ\/aGlpyX\/cCBAgQIAAgfIEChMa3d3d0dbWFj09PbnEkSNHYt68ebF\/\/\/6oq6sb0Vm8eHEcPXo0\/8lun\/3sZ2Pz5s15bJy7eemkvGFyLwIECBAg8G6BwoRGZ2dntLe3R1dXV36OJ06ciKampveERvbyypQpU+LFF1+Md955J+6444548MEHY8GCBULD\/iBAgAABAhUWKExoHD58OBYtWhTZSyjZ7cCBA7Fs2bKRZzjOud16661x\/fXXx\/Lly\/NfrVmzJv+EyrmXVLLfeUajwlNmOQIECBCoWoHChMbQ0FDMnDkztm\/fHrNnz87DoaamJjZs2BB9fX35R1lnzZoVzzzzTPzmN7\/Jjzt9+nTccsstsXbt2jw+zt2ERtXuBydOgAABAhUWKExoZC7ZyyYrVqzIiRoaGqKjoyOmTp2af6pkYGAgfy\/G22+\/Hffcc09knz7Jbp\/73Ofi+9\/\/vo+3VniwLEeAAAECBDKBQoVGdkLZx1yzqJgxY8YFr\/CxY8fyN4BmIfLum2c0bA4CBAgQIFAZgcKFRiVYhEYlFK1BgAABAgQK+IxGJS6q0KiEojUIECBAgIDQGHMGhIatQYAAAQIEKiPgpZMxHIVGZYbLKgQIECBAQGgIDbuAAAECBAgkExAaQiPZcFmYAAECBAgIDaFhFxAgQIAAgWQCQkNoJBsuCxMgQIAAAaEhNOwCAgQIECCQTEBoCI1kw2VhAgQIECAgNISGXUCAAAECBJIJCA2hkWy4LEyAAAECBISG0LALCBAgQIBAMgGhITSSDZeFCRAgQICA0BAadgEBAgQIEEgmIDSERrLhsjABAgQIEBAaQsMuIECAAAECyQSEhtBINlwWJkCAAAECQkNo2AUECBAgQCCZgNAQGsmGy8IECBAgQEBoCA27gAABAgQIJBMQGkIj2XBZmAABAgQICA2hYRcQIECAAIFkAkJDaCQbLgsTIECAAAGhITTsAgIECBAgkExAaAiNZMNlYQIECBAgUMjQGB4ejokTJ5a8uv\/+97\/jzJkzUVtbO+rYxsbG6O\/vL3l\/BxAgQIAAAQIXFihUaOzbty9WrVoVhw4dijlz5sSmTZuivr5+TIGhoaG47bbb4pOf\/GQ8\/PDDQsNOIUCAAAECCQQKExqnTp2K+fPnx9KlS6O5uTlaW1tj8uTJsX79+jHZsgh59tln45577hEaCQbLkgQIECBAIBMoTGj09vbGwoUL4+DBg\/lLIXv27MljY\/fu3e+50n\/4wx\/ikUceiaamprj88suFhr1AgAABAgQSCRQmNLq7u6OtrS16enpyqiNHjsS8efNi\/\/79UVdXN8L31ltvxY033hhbtmyJX\/7yl\/nvx3rpJPt9S0tL\/uNGgAABAgQIlCdQmNDo7OyM9vb26OrqyiVOnDiRP2Px7tBYvnx5XHXVVZH969NPP\/0fQ8ObQcsbKPciQIAAAQLnCxQmNA4fPhyLFi2K7CWU7HbgwIFYtmzZyDMc2e8GBwfzyMhuH\/rQh+Jf\/\/pX\/u9vuummeO6550ZcfOrEJiFAgAABApURKExoZJ8imTlzZmzfvj1mz54dq1evjpqamtiwYUP09fXlkTFr1qw4efLkiNzmzZsju9\/KlStHfcRVaFRmuKxCgAABAgQKExrZpcxeNlmxYkV+VRsaGqKjoyOmTp0a69ati4GBgcjC4vzbxo0b89Dw8VYbgQABAgQIpBEoVGhkRNnHXLOomDFjRtlintEom84dCRAgQIDAKIHChUYlrq\/QqISiNQgQIECAQIH+jkYlL6bQqKSmtQgQIECgmgU8ozHG1Rca1bwlnDsBAgQIVFJAaAiNSs6TtQgQIECAwCgBoSE0bAkCBAgQIJBMQGgIjWTDZWECBAgQICA0hIZdQIAAAQIEkgkIDaGRbLgsTIAAAQIEhIbQsAsIECBAgEAyAaEhNJINl4UJECBAgIDQEBp2AQECBAgQSCYgNIRGsuGyMAECBAgQEBpCwy4gQIAAAQLJBISG0Eg2XBYmQIAAAQJCQ2jYBQQIECBAIJmA0BAayYbLwgQIECBAQGgIDbuAAAECBAgkExAaQiPZcFmYAAECBAgIDaFhFxAgQIAAgWQCQkNoJBsuCxMgQIAAAaEhNOwCAgQIECCQTEBoCI1kw2VhAgQIECAgNISGXUCAAAECBJIJCA2hkWy4LEyAAAECBISG0LALCBAgQIBAMgGhITSSDZeFCRAgQICA0BAadgEBAgQIEEgmUMjQGB4ejokTJ14Q7fTp0\/kxNTU17zmusbEx+vv7k6FbmAABAgQIVItAoUJj3759sWrVqjh06FDMmTMnNm3aFPX19aOu5alTp2LlypWxa9eumDZtWnzlK1+JZcuWxeTJk0eOExrVMv7OkwABAgRSCxQmNLKAmD9\/fixdujSam5ujtbU1j4f169ePMty2bVv8+Mc\/zn8\/ZcqUWLRoUWzcuDFuuOEGoZF62qxPgAABAlUnUJjQ6O3tjYULF8bBgwejtrY29uzZk8fG7t27R13UNWvWxLXXXhs333xz\/vu77747Pvaxj8U3vvGNUaGR\/UNLS0v+40aAAAECBAiUJ1CY0Oju7o62trbo6enJJY4cORLz5s2L\/fv3R11d3Zg6v\/vd72LJkiWxc+fOuPrqqz2jUd4MuRcBAgQIEPiPAoUJjc7Ozmhvb4+urq78ZE+cOBFNTU1jhsZbb70VTz31VB4Y69aty19qOf\/mPRp2DAECBAgQqIxAYULj8OHD+fstspdQstuBAwfyN3mee4bjHNexY8fyZzGuu+66eOihh+KKK654j6TQqMxwWYUAAQIECBQmNIaGhmLmzJmxffv2mD17dqxevTr\/6OqGDRuir68vBgcHY9asWfHoo4\/GyZMn48knnxy5+pdddln+vo5zN6FhYxAgQIAAgcoIFCY0Mo7sZZMVK1bkMg0NDdHR0RFTp07NXx4ZGBiIzZs3x9y5c+Po0aOj9O64445Yu3at0KjMTFmFAAECBAiMCBQqNLKzyj7mmkXFjBkzyr7MntEom84dCRAgQIDAKIHChUYlrq\/QqISiNQgQIECAQITQGGMKhIatQYAAAQIEKiMgNIRGZSbJKgQIECBAYAwBoSE0bAwCBAgQIJBMQGgIjWTDZWECBAgQICA0hIZdQIAAAQIEkgkIDaGRbLgsTIAAAQIEhIbQsAsIECBAgEAyAaEhNJINl4UJECBAgIDQEBp2AQECBAgQSCYgNIRGsuGyMAECBAgQEBpCwy4gQIAAAQLJBISG0Eg2XBYmQIAAAQJCQ2jYBQQIECBAIJmA0BAayYbLwgQIECBAQGgIDbuAAAECBAgkExAaQiPZcFmYAAECBAgIDaFhFxAgQIAAgWQCQkNoJBsuCxMgQIAAAaEhNOwCAgQIECCQTEBoCI1kw2VhAgQIECAgNISGXUCAAAECBJIJCA2hkWy4LEyAAAECBISG0LALCBAgQIBAMgGhITSSDZeFCRAgQICA0BAadgEBAgQIEEgmULWhMTw8HBMnThwTtrGxMfr7+5OhW5gAAQIECFSLQNWFxpkzZ+Lxxx+Pn\/70p\/HBD34wHn744bj55ptHXW+hUS3j7zwJECBAILVA1YVGR0dHbNu2LV566aU4fvx4LFq0KH7729\/G9OnTR6yrJTSeeeaZaGlpST1j\/9P1q+Ec\/6fA\/ssJECBQQqDqQuOuu+6Ka665Ju6\/\/\/6c5otf\/GLcfffdceONN1ZdaFRDUFXDOfpfOQIECFzKAlUXGnPnzo2VK1fGF77whfy6PPTQQzFt2rT8X8\/dFi9eHHv37r2Ur5vHdpECTU1N8aMf\/egij3YYAQIECFRaoOpC41Of+lQ8\/fTTMWfOnNyytbU1pkyZEg8++GClba1HgAABAgSqXqDqQmP58uWRxUb2rEV2y142aW5ujs9\/\/vNVPwwACBAgQIBApQWqLjSyN4L++te\/zt8M2tvbm78ZdPfu3dHQ0FBpW+sRIECAAIGqF6i60Hj77bdjyZIl8ec\/\/zm\/+I888kj+rIYbAQIECBAgUHmBqguNc4R\/\/\/vfo76+Purq6iqvakUCBAgQIEAgF6ja0HD9CRAgQIAAgfQCQiO9sf8GAgQIECBQtQJCY4xLf6HvQRmPk3L27NnIfmpqasbjw\/+vH\/PQ0FBMmjTpv17HAgQIECDw\/gWExnlmF\/M9KO+f+H97j5dffjmyP8OdvQk2+06Xxx577D1fJverX\/0qvvWtb4080Msvvzz\/JE4Rbt3d3bF169bYsWNHEU7HORAgQGDcCQiN8y7ZxXwPyni6wn\/5y1\/yP62efYHcRz\/60bj11lvj3nvvjS996UujTuP555+P119\/Pb761a\/mv58wYcK4\/7hv9j02bW1t8corr8SVV14pNMbT4HqsBAgUSkBonHc5L+Z7UMbT1f\/hD38Yv\/jFL+InP\/lJ\/rA3btwYJ0+ejLVr1446jezPr2d\/qjv7Dpjsu0E+8IEPjKfTHPOxvvHGG9HT0xOvvvpq\/uMZjXF\/SZ0AAQLjVEBonHfhLuZ7UMbTdc5eDsmenXjiiSfyh93Z2Rnt7e3R1dU16jS+\/OUvx5\/+9Kf8WY8333wz7rvvvsL8bZFdu3ZF9oyN0BhPk+uxEiBQJAGhcd7VLNr3oGR\/jOyKK64Y+cK4nTt3xg9+8IP42c9+NmqGs\/i47rrr4uqrr86\/TC778+y\/\/\/3v8\/uO95vQGO9X0OMnQGC8CwiN865g0b4HJfsz61k4PPfcc\/lZbt++PV577bV4\/PHHR8769OnTkb2fYfr06SO\/u\/baa\/NnAc598dx4HnKhMZ6vnsdOgEARBITGeVexaN+Dkv2Z9dtvvz1\/Q2T2kd3sPSh33nln\/umTvr6+GBwcjKuuuiquv\/76+OY3v5m\/STT7HpgHHngg\/vjHPxbivRpCowj\/M+UcCBAYzwJC47yrV7TvQcn+dsaaNWvixRdfzM9ywYIF8eyzz+Z\/U2LdunUxMDAQmzdvjuz\/jLPjslv26ZMnn3wy\/4RKEW7Zy0VbtmzxHo0iXEznQIDAuBQQGmNctqJ9D8o\/\/\/nPyP5GyEc+8pH\/OKTZMx5\/+9vfYtq0aVFbWzsuh9mDJkCAAIFLT0BoXHrXxCMiQIAAAQKFERAahbmUToQAAQIECFx6AkLj0rsmHhEBAgQIECiMgNAozKV0IgQIECBA4NITEBqX3jXxiAgQIECAQGEEhEZhLqUTIUCAAAECl56A0Lj0rolHRIAAAQIECiPwf+JRJfdDe4CeAAAAAElFTkSuQmCC","height":224,"width":538}}
%---
%[output:6271552f]
%   data: {"dataType":"error","outputData":{"errorType":"runtime","text":"Error using <a href=\"matlab:matlab.lang.internal.introspective.errorDocCallback('bar', '\/Applications\/MATLAB_R2025a.app\/toolbox\/matlab\/graphics\/graphics\/specgraph\/bar.m', 176)\" style=\"font-weight:bold\">bar<\/a> (<a href=\"matlab: opentoline('\/Applications\/MATLAB_R2025a.app\/toolbox\/matlab\/graphics\/graphics\/specgraph\/bar.m',176,0)\">line 176<\/a>)\nThe length of all vector inputs must match either the number of rows or the number of columns of all 2D matrix inputs.\n\nError in <a href=\"matlab:matlab.lang.internal.introspective.errorDocCallback('Project_Practice_Area_Expansion>plotCorrelationResults', '\/Users\/cobybowman\/Documents\/MATLAB\/Examples\/R2022b\/matlab\/Davis_Lab\/Project_Practice_Area_Expansion.mlx', 930)\" style=\"font-weight:bold\">Project_Practice_Area_Expansion>plotCorrelationResults<\/a> (<a href=\"matlab: opentoline('\/Users\/cobybowman\/Documents\/MATLAB\/Examples\/R2022b\/matlab\/Davis_Lab\/Project_Practice_Area_Expansion.mlx',930,0)\">line 930<\/a>)\n    bar(lags_ms, results.acg{bursty_idx}, 'k');\n    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nError in <a href=\"matlab:matlab.lang.internal.introspective.errorDocCallback('Project_Practice_Area_Expansion>runSpikeCorrelationAnalysis', '\/Users\/cobybowman\/Documents\/MATLAB\/Examples\/R2022b\/matlab\/Davis_Lab\/Project_Practice_Area_Expansion.mlx', 809)\" style=\"font-weight:bold\">Project_Practice_Area_Expansion>runSpikeCorrelationAnalysis<\/a> (<a href=\"matlab: opentoline('\/Users\/cobybowman\/Documents\/MATLAB\/Examples\/R2022b\/matlab\/Davis_Lab\/Project_Practice_Area_Expansion.mlx',809,0)\">line 809<\/a>)\n    correlationResults.figures = plotCorrelationResults(correlationResults, isExc, isEnh);\n    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"}}
%---
