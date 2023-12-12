This repository includes the source code of the STA-DNN [1] and TRFI-DNN [2] channel estimators proposed in "Deep Learning Based Channel Estimation Schemes for IEEE 802.11 p Standard" and "Joint TRFI and Deep Learning for Vehicular Channel Estimation" papers that are published in the IEEE Access journal and the proceedings of the 2020 IEEE GLOBECOM  Workshops, respectively. Please note that the Tx-Rx OFDM processing is implemented in Matlab (Matlab_Codes) and the LSTM processing is implemented in python (PyTorch) (Python_Codes).


### Files Description 
- Main.m: The main simulation file, where the simulation parameters (Channel model, OFDM parameters, Modulation scheme, etc...) are defined. 
- Channel_functions.m: Includes the pre-defined vehicular channel models [3] for different mobility conditions.
- Initial_Channel_Estimation.m: Includes the implementation of the data-pilot aided (DPA) channel estimation. Please note that DPA is the initial            estimation step in the following conventional channel estimators.
- STA.m: Includes the implementation of the spectral temporal averaging (STA) channel estimator [4].
- CDP.m: Includes the implementation of the constructed data pilots (CDP) channel estimator [5].
- TRFI.m: Includes the implementation of the time-domain reliability test frequency-domain interpolation (TRFI) channel estimator [6].
- MMSE_Virtual_Pilots.m: Includes the implementation of the minimum mean square error using virtual pilots (MMSE-VP) channel estimator [7].
- DNN_Datasets_Generation.m: Complex to real values conversions of the generated datasets for the studied channel estimation scheme. 
- DNN_Results_Processing.m: Processing the testing results genertead by the DNN_Testing file and caculate the BER and NMSE results of the DNN-based estimators.
- DNN.py: The DNN training/testing is performed employing the generated training/testing datasets. The file should be executed twice as follows:
	- **Step1: Training by executing this command python DNN.py  Mobility Channel_Model Modulation_Order Channel_Estimator Training_SNR DNN_Input Hidden_Layer1 Hidden_Layer2 Hidden_Layer3 DNN_Output Epochs Batch_size**
	- **Step2: Testing by executing this command: python DNN.py  Mobility Channel_Model Modulation_Scheme Channel_Estimator Testing_SNR** 
> ex: python DNN.py  High VTV_SDWW QPSK STA 40 104 15 10 15 104 500 128

> ex: python DNN.py High VTV_SDWW QPSK STA 40
		
### Running Steps:
1. Run the IDX_Generation.m in order to genertae the dataset indices, training dataset size, and testing dataset size.
2. Run the main.m file two times as follows:
	- Specify all the simulation parameters like: the number of OFDM symbols, channel model, mobility scenario, modulatio order, SNR range, etc.
	- Specify the path of the generated indices in step (1).
	- The first time for generating the traininig simulation file (set the configuration = 'training' in the code).
	- The second time for generating the testing simulations files (set the configuration = 'testing' in the code).
	- After that, the generated simulations files will be saved in your working directory.
3. Run the DNN_Datasets_Generation.m also two times by changing the configuration as done in step (2) in addition to specifying the channel estimation scheme as well as the OFDM simulation parameters. This step convert the generated datasets from complex to real values.
4. Run the DNN.py file also two times in order to perform the training first then the testing as mentioned in the DNN.py file description.
5. After finishing step 4, the DNN results will be saved as a .mat files. Then you need to run the DNN_Results_Processing.m file in order to get the NMSE and BER results of the studied channel estimation scheme.

### References
- [1] A. K. Gizzini, M. Chafii, A. Nimr and G. Fettweis, "Deep Learning Based Channel Estimation Schemes for IEEE 802.11p Standard," in IEEE Access, vol. 8, pp. 113751-113765, 2020.
- [2] A. K. Gizzini, M. Chafii, A. Nimr and G. Fettweis, "Joint TRFI and Deep Learning for Vehicular Channel Estimation," 2020 IEEE Globecom Workshops (GC Wkshps, Taipei, Taiwan, 2020, pp. 1-6.
- [3] G. Acosta-Marum and M. A. Ingram, ‘‘Six time- and frequency-selective empirical channel models for vehicular wireless LANs,’’ IEEE Veh. Technol. Mag., vol. 2, no. 4, pp. 4–11, Dec. 2007.
- [4] J. A. Fernandez, K. Borries, L. Cheng, B. V. K. V. Kumar, D. D. Stancil, and F. Bai, ‘‘Performance of the 802.11p physical layer in vehicle-tovehicle environments,’’ IEEE Trans. Veh. Technol., vol. 61, no. 1, pp. 3–14, Jan. 2012.
- [5] Z. Zhao, X. Cheng, M. Wen, B. Jiao, and C.-X. Wang, ‘‘Channel estimation schemes for IEEE 802.11p standard,’’ IEEE Intell. Transp. Syst. Mag., vol. 5, no. 4, pp. 38–49, Oct. 2013.
- [6] Y.-K. Kim, J.-M. Oh, J.-H. Shin, and C. Mun, ‘‘Time and frequency domain channel estimation scheme for IEEE 802.11p,’’ in Proc. 17th Int. 
IEEE Conf. Intell. Transp. Syst. (ITSC), Oct. 2014, pp. 1085–1090.
- [7] J.-Y. Choi, K.-H. Yoo, and C. Mun, ‘‘MMSE channel estimation scheme using virtual pilot signal for IEEE 802.11p,’’ J. Korean Inst. Inf. Technol., vol. 27, no. 1, pp. 27–32, Jan. 2016. 


For more information and questions, please contact me on abdulkarimgizzini@gmail.com
