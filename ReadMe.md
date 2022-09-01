# Update in progress

# This project demonstartes:
- The IEEE 802.11p stanadrd physical layer Tx-Rx implementation with several conventional channel estimators.
- The full implementation of the STA-DNNs and TRFI-DNN channel estimators proposed in [1] and [2] respectively.
- Note: the project's files are divided into Matlab_Codes and Python_Codes.
	- The Matlab_Codes folder includes all the required functions to simulate the Tx-Rx OFDM communication.
	- The Python_Codes folder includes the implementation of the deep neural network (DNN) training and testing phases. 

### Matlab_Codes
- Main.m: The main simulation file, where the simulation parameters (Channel model, OFDM parameters, Modulation scheme, etc...) are defined. 
- Channel_functions.m: Includes the pre-defined vehicular channel models[3] for different mobility conditions.
- Initial_Channel_Estimation.m: Includes the implementation of the data-pilot aided (DPA) channel estimation. Please note that DPA is the initial            estimation step in the following conventional channel estimators.
- STA.m: Includes the implementation of the spectral temporal averaging (STA) channel estimator [4].
- CDP.m: Includes the implementation of the constructed data pilots (CDP) channel estimator [5].
- TRFI.m: Includes the implementation of the time-domain reliability test frequency-domain interpolation (TRFI) channel estimator [6].
- MMSE_Virtual_Pilots.m: Includes the implementation of the minimum mean square error using virtual pilots (MMSE-VP) channel estimator [7].
- DNN_Datasets_Generation.m: Divides the saved estimated channels (ex: STA_Structure) into training and testing datasets. Forexample, in the project 	       we consider (1000 Channel realiztions with nSymbols = 100 per frame), then the saved STA_Structure will be of size 52x100x1000. Therefore we                 select 800 and 200 random indices from (1000 indices) to be the training and testing OFDM frames indices, respectively. (Since 80% of data is for           training and 20% of data is for testing). The generated datasets will be processed by the Python_Codes files as explained below. 
- DNN_Results_Processing.m: Processing the testing results genertead by the DNN_Testing file and caculate the BER and NMSE results of the DNN-based estimators.
		
		
### Python_Codes
- DNN_Training.py: here the DNN training is performed employing the training datasets and the simulation parameters using the following terminal 	   command: 
	- **python DNN_Training.py Channel_Model Modulation_Order Channel_Estimator Training_SNR DNN_Input Hidden_Layer1 Hidden_Layer2 Hidden_Layer3 DNN_Output Epochs Batch_size** 
> ex: python DNN_Training.py VTV_UC QPSK STA 7 104 15 10 15 104 500 128
- DNN_Testing.py: The testing datasets are tested using the trained DNN model, and the testing results are saved in .mat files, in order to be processed later. The DNN testing can be executed similarly as performed in the training:
	- **python DNN_Testing.py Channel_Model Modulation_Scheme Channel_Estimator Hidden_Layer1 Hidden_Layer2 Hidden_Layer3** 
> ex: python DNN_Testing.py VTV_UC QPSK STA 15 10 15
		
For more information and questions, please contact me on abdulkarim.gizzini@ensea.fr


### Proposed Methodology

### Performance Evaluation 

### Computational Complexity Analysis 

### References
- [1] A. K. Gizzini, M. Chafii, A. Nimr and G. Fettweis, "Deep Learning Based Channel Estimation Schemes for IEEE 802.11p Standard," in IEEE Access, vol. 8, pp. 113751-113765, 2020.
- [2] A. K. Gizzini, M. Chafii, A. Nimr and G. Fettweis, "Joint TRFI and Deep Learning for Vehicular Channel Estimation," 2020 IEEE Globecom Workshops (GC Wkshps, Taipei, Taiwan, 2020, pp. 1-6.
- [3] G. Acosta-Marum and M. A. Ingram, ‘‘Six time- and frequency-selective empirical channel models for vehicular wireless LANs,’’ IEEE Veh. Technol. Mag., vol. 2, no. 4, pp. 4–11, Dec. 2007.
- [4] J. A. Fernandez, K. Borries, L. Cheng, B. V. K. V. Kumar, D. D. Stancil, and F. Bai, ‘‘Performance of the 802.11p physical layer in vehicle-tovehicle environments,’’ IEEE Trans. Veh. Technol., vol. 61, no. 1, pp. 3–14, Jan. 2012.
- [5] Z. Zhao, X. Cheng, M. Wen, B. Jiao, and C.-X. Wang, ‘‘Channel estimation schemes for IEEE 802.11p standard,’’ IEEE Intell. Transp. Syst. Mag., vol. 5, no. 4, pp. 38–49, Oct. 2013.
- [6] Y.-K. Kim, J.-M. Oh, J.-H. Shin, and C. Mun, ‘‘Time and frequency domain channel estimation scheme for IEEE 802.11p,’’ in Proc. 17th Int. 
IEEE Conf. Intell. Transp. Syst. (ITSC), Oct. 2014, pp. 1085–1090.
- [7] J.-Y. Choi, K.-H. Yoo, and C. Mun, ‘‘MMSE channel estimation scheme using virtual pilot signal for IEEE 802.11p,’’ J. Korean Inst. Inf. Technol., vol. 27, no. 1, pp. 27–32, Jan. 2016.


### Citations
