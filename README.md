# RUL-prognosis

### NASA Commercial Modular Aero-Propulsion System (C-MAPSS) turbofan engine data set.

This dataset is used to be available at NASA's Prognostics Center of Execellence Data Repository (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). C-MAPSS simulates an engine model of the 90,000 lb thrust class and the package includes an atmospheric model capable of simulating operations at
(i) different altitudes, 
(ii) Mach numbers 
(iii) sea-level temperatures

The data produced was divided into training and test sets. Each series in the train set runs until system failure while the test set was terminated some time before any faults. The target is the number of remaining cycles before failure. The training set had trajectories that ended at the failure threshold while the test and validation sets were pruned to stop some time prior to the failure threshold. The outputs include various sensor response surfaces and operability margins: 21 variables out of 58 different outputs available from the model.
