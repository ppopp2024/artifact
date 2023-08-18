extern __shared__ float shared[];
extern "C" __global__ void __launch_bounds__(256) VkFFT_main_logN25_2 (float2* inputs, float2* outputs) {
unsigned int sharedStride = 16;
float2* sdata = (float2*)shared;

	float2 temp_0;
	temp_0.x=0;
	temp_0.y=0;
	float2 temp_1;
	temp_1.x=0;
	temp_1.y=0;
	float2 temp_2;
	temp_2.x=0;
	temp_2.y=0;
	float2 temp_3;
	temp_3.x=0;
	temp_3.y=0;
	float2 temp_4;
	temp_4.x=0;
	temp_4.y=0;
	float2 temp_5;
	temp_5.x=0;
	temp_5.y=0;
	float2 temp_6;
	temp_6.x=0;
	temp_6.y=0;
	float2 temp_7;
	temp_7.x=0;
	temp_7.y=0;
	float2 temp_8;
	temp_8.x=0;
	temp_8.y=0;
	float2 temp_9;
	temp_9.x=0;
	temp_9.y=0;
	float2 temp_10;
	temp_10.x=0;
	temp_10.y=0;
	float2 temp_11;
	temp_11.x=0;
	temp_11.y=0;
	float2 temp_12;
	temp_12.x=0;
	temp_12.y=0;
	float2 temp_13;
	temp_13.x=0;
	temp_13.y=0;
	float2 temp_14;
	temp_14.x=0;
	temp_14.y=0;
	float2 temp_15;
	temp_15.x=0;
	temp_15.y=0;
	float2 w;
	w.x=0;
	w.y=0;
	float2 loc_0;
	loc_0.x=0;
	loc_0.y=0;
	float2 iw;
	iw.x=0;
	iw.y=0;
	unsigned int stageInvocationID=0;
	unsigned int blockInvocationID=0;
	unsigned int sdataID=0;
	unsigned int combinedID=0;
	unsigned int inoutID=0;
	unsigned int disableThreads=1;
	float angle=0;
	float2 mult;
	mult.x = 0;
	mult.y = 0;
		disableThreads = ((((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072) < 33554432) ? 1 : 0;
		if(disableThreads>0) {
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 0) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_0=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 16) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_1=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 32) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_2=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 48) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_3=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 64) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_4=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 80) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_5=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 96) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_6=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 112) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_7=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 128) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_8=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 144) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_9=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 160) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_10=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 176) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_11=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 192) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_12=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 208) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_13=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 224) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
			temp_14=inputs[inoutID];
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 240) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID);
		if(threadIdx.y < 16){
			temp_15=inputs[inoutID];
		}
	}
		if(disableThreads>0) {
		stageInvocationID = (threadIdx.y+ 0) % (1);
		angle = stageInvocationID * -3.14159265358979312e+00f;
	w.x = 1;
	w.y = 0;
	loc_0.x = temp_8.x * w.x - temp_8.y * w.y;
	loc_0.y = temp_8.y * w.x + temp_8.x * w.y;
	temp_8.x = temp_0.x - loc_0.x;
	temp_8.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_9.x * w.x - temp_9.y * w.y;
	loc_0.y = temp_9.y * w.x + temp_9.x * w.y;
	temp_9.x = temp_1.x - loc_0.x;
	temp_9.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	loc_0.x = temp_10.x * w.x - temp_10.y * w.y;
	loc_0.y = temp_10.y * w.x + temp_10.x * w.y;
	temp_10.x = temp_2.x - loc_0.x;
	temp_10.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	loc_0.x = temp_11.x * w.x - temp_11.y * w.y;
	loc_0.y = temp_11.y * w.x + temp_11.x * w.y;
	temp_11.x = temp_3.x - loc_0.x;
	temp_11.y = temp_3.y - loc_0.y;
	temp_3.x = temp_3.x + loc_0.x;
	temp_3.y = temp_3.y + loc_0.y;
	loc_0.x = temp_12.x * w.x - temp_12.y * w.y;
	loc_0.y = temp_12.y * w.x + temp_12.x * w.y;
	temp_12.x = temp_4.x - loc_0.x;
	temp_12.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	loc_0.x = temp_13.x * w.x - temp_13.y * w.y;
	loc_0.y = temp_13.y * w.x + temp_13.x * w.y;
	temp_13.x = temp_5.x - loc_0.x;
	temp_13.y = temp_5.y - loc_0.y;
	temp_5.x = temp_5.x + loc_0.x;
	temp_5.y = temp_5.y + loc_0.y;
	loc_0.x = temp_14.x * w.x - temp_14.y * w.y;
	loc_0.y = temp_14.y * w.x + temp_14.x * w.y;
	temp_14.x = temp_6.x - loc_0.x;
	temp_14.y = temp_6.y - loc_0.y;
	temp_6.x = temp_6.x + loc_0.x;
	temp_6.y = temp_6.y + loc_0.y;
	loc_0.x = temp_15.x * w.x - temp_15.y * w.y;
	loc_0.y = temp_15.y * w.x + temp_15.x * w.y;
	temp_15.x = temp_7.x - loc_0.x;
	temp_15.y = temp_7.y - loc_0.y;
	temp_7.x = temp_7.x + loc_0.x;
	temp_7.y = temp_7.y + loc_0.y;
	w.x = 1;
	w.y = 0;
	loc_0.x = temp_4.x * w.x - temp_4.y * w.y;
	loc_0.y = temp_4.y * w.x + temp_4.x * w.y;
	temp_4.x = temp_0.x - loc_0.x;
	temp_4.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_5.x * w.x - temp_5.y * w.y;
	loc_0.y = temp_5.y * w.x + temp_5.x * w.y;
	temp_5.x = temp_1.x - loc_0.x;
	temp_5.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	loc_0.x = temp_6.x * w.x - temp_6.y * w.y;
	loc_0.y = temp_6.y * w.x + temp_6.x * w.y;
	temp_6.x = temp_2.x - loc_0.x;
	temp_6.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_3.x - loc_0.x;
	temp_7.y = temp_3.y - loc_0.y;
	temp_3.x = temp_3.x + loc_0.x;
	temp_3.y = temp_3.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_12.x * iw.x - temp_12.y * iw.y;
	loc_0.y = temp_12.y * iw.x + temp_12.x * iw.y;
	temp_12.x = temp_8.x - loc_0.x;
	temp_12.y = temp_8.y - loc_0.y;
	temp_8.x = temp_8.x + loc_0.x;
	temp_8.y = temp_8.y + loc_0.y;
	loc_0.x = temp_13.x * iw.x - temp_13.y * iw.y;
	loc_0.y = temp_13.y * iw.x + temp_13.x * iw.y;
	temp_13.x = temp_9.x - loc_0.x;
	temp_13.y = temp_9.y - loc_0.y;
	temp_9.x = temp_9.x + loc_0.x;
	temp_9.y = temp_9.y + loc_0.y;
	loc_0.x = temp_14.x * iw.x - temp_14.y * iw.y;
	loc_0.y = temp_14.y * iw.x + temp_14.x * iw.y;
	temp_14.x = temp_10.x - loc_0.x;
	temp_14.y = temp_10.y - loc_0.y;
	temp_10.x = temp_10.x + loc_0.x;
	temp_10.y = temp_10.y + loc_0.y;
	loc_0.x = temp_15.x * iw.x - temp_15.y * iw.y;
	loc_0.y = temp_15.y * iw.x + temp_15.x * iw.y;
	temp_15.x = temp_11.x - loc_0.x;
	temp_15.y = temp_11.y - loc_0.y;
	temp_11.x = temp_11.x + loc_0.x;
	temp_11.y = temp_11.y + loc_0.y;
	w.x = 1;
	w.y = 0;
	loc_0.x = temp_2.x * w.x - temp_2.y * w.y;
	loc_0.y = temp_2.y * w.x + temp_2.x * w.y;
	temp_2.x = temp_0.x - loc_0.x;
	temp_2.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_3.x * w.x - temp_3.y * w.y;
	loc_0.y = temp_3.y * w.x + temp_3.x * w.y;
	temp_3.x = temp_1.x - loc_0.x;
	temp_3.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_6.x * iw.x - temp_6.y * iw.y;
	loc_0.y = temp_6.y * iw.x + temp_6.x * iw.y;
	temp_6.x = temp_4.x - loc_0.x;
	temp_6.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	loc_0.x = temp_7.x * iw.x - temp_7.y * iw.y;
	loc_0.y = temp_7.y * iw.x + temp_7.x * iw.y;
	temp_7.x = temp_5.x - loc_0.x;
	temp_7.y = temp_5.y - loc_0.y;
	temp_5.x = temp_5.x + loc_0.x;
	temp_5.y = temp_5.y + loc_0.y;
	iw.x = w.x * 7.07106781186547573e-01f + w.y * 7.07106781186547573e-01f;
	iw.y = w.y * 7.07106781186547573e-01f - w.x * 7.07106781186547573e-01f;

	loc_0.x = temp_10.x * iw.x - temp_10.y * iw.y;
	loc_0.y = temp_10.y * iw.x + temp_10.x * iw.y;
	temp_10.x = temp_8.x - loc_0.x;
	temp_10.y = temp_8.y - loc_0.y;
	temp_8.x = temp_8.x + loc_0.x;
	temp_8.y = temp_8.y + loc_0.y;
	loc_0.x = temp_11.x * iw.x - temp_11.y * iw.y;
	loc_0.y = temp_11.y * iw.x + temp_11.x * iw.y;
	temp_11.x = temp_9.x - loc_0.x;
	temp_11.y = temp_9.y - loc_0.y;
	temp_9.x = temp_9.x + loc_0.x;
	temp_9.y = temp_9.y + loc_0.y;
	w.x = iw.y;
	w.y = -iw.x;
	loc_0.x = temp_14.x * w.x - temp_14.y * w.y;
	loc_0.y = temp_14.y * w.x + temp_14.x * w.y;
	temp_14.x = temp_12.x - loc_0.x;
	temp_14.y = temp_12.y - loc_0.y;
	temp_12.x = temp_12.x + loc_0.x;
	temp_12.y = temp_12.y + loc_0.y;
	loc_0.x = temp_15.x * w.x - temp_15.y * w.y;
	loc_0.y = temp_15.y * w.x + temp_15.x * w.y;
	temp_15.x = temp_13.x - loc_0.x;
	temp_15.y = temp_13.y - loc_0.y;
	temp_13.x = temp_13.x + loc_0.x;
	temp_13.y = temp_13.y + loc_0.y;
	w.x = 1;
	w.y = 0;
	loc_0.x = temp_1.x * w.x - temp_1.y * w.y;
	loc_0.y = temp_1.y * w.x + temp_1.x * w.y;
	temp_1.x = temp_0.x - loc_0.x;
	temp_1.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_3.x * iw.x - temp_3.y * iw.y;
	loc_0.y = temp_3.y * iw.x + temp_3.x * iw.y;
	temp_3.x = temp_2.x - loc_0.x;
	temp_3.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	iw.x = w.x * 7.07106781186547573e-01f + w.y * 7.07106781186547573e-01f;
	iw.y = w.y * 7.07106781186547573e-01f - w.x * 7.07106781186547573e-01f;

	loc_0.x = temp_5.x * iw.x - temp_5.y * iw.y;
	loc_0.y = temp_5.y * iw.x + temp_5.x * iw.y;
	temp_5.x = temp_4.x - loc_0.x;
	temp_5.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	loc_0.x = iw.y;
	loc_0.y = -iw.x;
	iw = loc_0;
	loc_0.x = temp_7.x * iw.x - temp_7.y * iw.y;
	loc_0.y = temp_7.y * iw.x + temp_7.x * iw.y;
	temp_7.x = temp_6.x - loc_0.x;
	temp_7.y = temp_6.y - loc_0.y;
	temp_6.x = temp_6.x + loc_0.x;
	temp_6.y = temp_6.y + loc_0.y;
	iw.x = w.x * 9.23879532511286738e-01f + w.y * 3.82683432365089782e-01f;
	iw.y = w.y * 9.23879532511286738e-01f - w.x * 3.82683432365089782e-01f;

	loc_0.x = temp_9.x * iw.x - temp_9.y * iw.y;
	loc_0.y = temp_9.y * iw.x + temp_9.x * iw.y;
	temp_9.x = temp_8.x - loc_0.x;
	temp_9.y = temp_8.y - loc_0.y;
	temp_8.x = temp_8.x + loc_0.x;
	temp_8.y = temp_8.y + loc_0.y;
	loc_0.x = iw.y;
	loc_0.y = -iw.x;
	iw = loc_0;
	loc_0.x = temp_11.x * iw.x - temp_11.y * iw.y;
	loc_0.y = temp_11.y * iw.x + temp_11.x * iw.y;
	temp_11.x = temp_10.x - loc_0.x;
	temp_11.y = temp_10.y - loc_0.y;
	temp_10.x = temp_10.x + loc_0.x;
	temp_10.y = temp_10.y + loc_0.y;
	iw.x = w.x * 3.82683432365089837e-01f + w.y * 9.23879532511286738e-01f;
	iw.y = w.y * 3.82683432365089837e-01f - w.x * 9.23879532511286738e-01f;

	loc_0.x = temp_13.x * iw.x - temp_13.y * iw.y;
	loc_0.y = temp_13.y * iw.x + temp_13.x * iw.y;
	temp_13.x = temp_12.x - loc_0.x;
	temp_13.y = temp_12.y - loc_0.y;
	temp_12.x = temp_12.x + loc_0.x;
	temp_12.y = temp_12.y + loc_0.y;
	loc_0.x = iw.y;
	loc_0.y = -iw.x;
	iw = loc_0;
	loc_0.x = temp_15.x * iw.x - temp_15.y * iw.y;
	loc_0.y = temp_15.y * iw.x + temp_15.x * iw.y;
	temp_15.x = temp_14.x - loc_0.x;
	temp_15.y = temp_14.y - loc_0.y;
	temp_14.x = temp_14.x + loc_0.x;
	temp_14.y = temp_14.y + loc_0.y;
}		sharedStride = 16;
	__syncthreads();

		if(disableThreads>0) {
	stageInvocationID = threadIdx.y + 0;
	blockInvocationID = stageInvocationID;
	stageInvocationID = stageInvocationID % 1;
	blockInvocationID = blockInvocationID - stageInvocationID;
	inoutID = blockInvocationID * 16;
	inoutID = inoutID + stageInvocationID;
	sdataID = inoutID + 0;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_0;
	sdataID = inoutID + 1;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_8;
	sdataID = inoutID + 2;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_4;
	sdataID = inoutID + 3;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_12;
	sdataID = inoutID + 4;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_2;
	sdataID = inoutID + 5;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_10;
	sdataID = inoutID + 6;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_6;
	sdataID = inoutID + 7;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_14;
	sdataID = inoutID + 8;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_1;
	sdataID = inoutID + 9;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_9;
	sdataID = inoutID + 10;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_5;
	sdataID = inoutID + 11;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_13;
	sdataID = inoutID + 12;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_3;
	sdataID = inoutID + 13;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_11;
	sdataID = inoutID + 14;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_7;
	sdataID = inoutID + 15;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_15;
}	__syncthreads();

		if(disableThreads>0) {
		stageInvocationID = (threadIdx.y+ 0) % (16);
		angle = stageInvocationID * -1.96349540849362070e-01f;
		temp_0 = sdata[sharedStride*(threadIdx.y+0)+threadIdx.x];
		temp_8 = sdata[sharedStride*(threadIdx.y+16)+threadIdx.x];
		temp_4 = sdata[sharedStride*(threadIdx.y+32)+threadIdx.x];
		temp_12 = sdata[sharedStride*(threadIdx.y+48)+threadIdx.x];
		temp_2 = sdata[sharedStride*(threadIdx.y+64)+threadIdx.x];
		temp_10 = sdata[sharedStride*(threadIdx.y+80)+threadIdx.x];
		temp_6 = sdata[sharedStride*(threadIdx.y+96)+threadIdx.x];
		temp_14 = sdata[sharedStride*(threadIdx.y+112)+threadIdx.x];
		temp_1 = sdata[sharedStride*(threadIdx.y+128)+threadIdx.x];
		temp_9 = sdata[sharedStride*(threadIdx.y+144)+threadIdx.x];
		temp_5 = sdata[sharedStride*(threadIdx.y+160)+threadIdx.x];
		temp_13 = sdata[sharedStride*(threadIdx.y+176)+threadIdx.x];
		temp_3 = sdata[sharedStride*(threadIdx.y+192)+threadIdx.x];
		temp_11 = sdata[sharedStride*(threadIdx.y+208)+threadIdx.x];
		temp_7 = sdata[sharedStride*(threadIdx.y+224)+threadIdx.x];
		temp_15 = sdata[sharedStride*(threadIdx.y+240)+threadIdx.x];
	w.x = __cosf(angle);
	w.y = __sinf(angle);
	loc_0.x = temp_1.x * w.x - temp_1.y * w.y;
	loc_0.y = temp_1.y * w.x + temp_1.x * w.y;
	temp_1.x = temp_0.x - loc_0.x;
	temp_1.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_9.x * w.x - temp_9.y * w.y;
	loc_0.y = temp_9.y * w.x + temp_9.x * w.y;
	temp_9.x = temp_8.x - loc_0.x;
	temp_9.y = temp_8.y - loc_0.y;
	temp_8.x = temp_8.x + loc_0.x;
	temp_8.y = temp_8.y + loc_0.y;
	loc_0.x = temp_5.x * w.x - temp_5.y * w.y;
	loc_0.y = temp_5.y * w.x + temp_5.x * w.y;
	temp_5.x = temp_4.x - loc_0.x;
	temp_5.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	loc_0.x = temp_13.x * w.x - temp_13.y * w.y;
	loc_0.y = temp_13.y * w.x + temp_13.x * w.y;
	temp_13.x = temp_12.x - loc_0.x;
	temp_13.y = temp_12.y - loc_0.y;
	temp_12.x = temp_12.x + loc_0.x;
	temp_12.y = temp_12.y + loc_0.y;
	loc_0.x = temp_3.x * w.x - temp_3.y * w.y;
	loc_0.y = temp_3.y * w.x + temp_3.x * w.y;
	temp_3.x = temp_2.x - loc_0.x;
	temp_3.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	loc_0.x = temp_11.x * w.x - temp_11.y * w.y;
	loc_0.y = temp_11.y * w.x + temp_11.x * w.y;
	temp_11.x = temp_10.x - loc_0.x;
	temp_11.y = temp_10.y - loc_0.y;
	temp_10.x = temp_10.x + loc_0.x;
	temp_10.y = temp_10.y + loc_0.y;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_6.x - loc_0.x;
	temp_7.y = temp_6.y - loc_0.y;
	temp_6.x = temp_6.x + loc_0.x;
	temp_6.y = temp_6.y + loc_0.y;
	loc_0.x = temp_15.x * w.x - temp_15.y * w.y;
	loc_0.y = temp_15.y * w.x + temp_15.x * w.y;
	temp_15.x = temp_14.x - loc_0.x;
	temp_15.y = temp_14.y - loc_0.y;
	temp_14.x = temp_14.x + loc_0.x;
	temp_14.y = temp_14.y + loc_0.y;
	w.x = __cosf(0.5f*angle);
	w.y = __sinf(0.5f*angle);
	loc_0.x = temp_2.x * w.x - temp_2.y * w.y;
	loc_0.y = temp_2.y * w.x + temp_2.x * w.y;
	temp_2.x = temp_0.x - loc_0.x;
	temp_2.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_10.x * w.x - temp_10.y * w.y;
	loc_0.y = temp_10.y * w.x + temp_10.x * w.y;
	temp_10.x = temp_8.x - loc_0.x;
	temp_10.y = temp_8.y - loc_0.y;
	temp_8.x = temp_8.x + loc_0.x;
	temp_8.y = temp_8.y + loc_0.y;
	loc_0.x = temp_6.x * w.x - temp_6.y * w.y;
	loc_0.y = temp_6.y * w.x + temp_6.x * w.y;
	temp_6.x = temp_4.x - loc_0.x;
	temp_6.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	loc_0.x = temp_14.x * w.x - temp_14.y * w.y;
	loc_0.y = temp_14.y * w.x + temp_14.x * w.y;
	temp_14.x = temp_12.x - loc_0.x;
	temp_14.y = temp_12.y - loc_0.y;
	temp_12.x = temp_12.x + loc_0.x;
	temp_12.y = temp_12.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_3.x * iw.x - temp_3.y * iw.y;
	loc_0.y = temp_3.y * iw.x + temp_3.x * iw.y;
	temp_3.x = temp_1.x - loc_0.x;
	temp_3.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	loc_0.x = temp_11.x * iw.x - temp_11.y * iw.y;
	loc_0.y = temp_11.y * iw.x + temp_11.x * iw.y;
	temp_11.x = temp_9.x - loc_0.x;
	temp_11.y = temp_9.y - loc_0.y;
	temp_9.x = temp_9.x + loc_0.x;
	temp_9.y = temp_9.y + loc_0.y;
	loc_0.x = temp_7.x * iw.x - temp_7.y * iw.y;
	loc_0.y = temp_7.y * iw.x + temp_7.x * iw.y;
	temp_7.x = temp_5.x - loc_0.x;
	temp_7.y = temp_5.y - loc_0.y;
	temp_5.x = temp_5.x + loc_0.x;
	temp_5.y = temp_5.y + loc_0.y;
	loc_0.x = temp_15.x * iw.x - temp_15.y * iw.y;
	loc_0.y = temp_15.y * iw.x + temp_15.x * iw.y;
	temp_15.x = temp_13.x - loc_0.x;
	temp_15.y = temp_13.y - loc_0.y;
	temp_13.x = temp_13.x + loc_0.x;
	temp_13.y = temp_13.y + loc_0.y;
	w.x = __cosf(0.25f*angle);
	w.y = __sinf(0.25f*angle);
	loc_0.x = temp_4.x * w.x - temp_4.y * w.y;
	loc_0.y = temp_4.y * w.x + temp_4.x * w.y;
	temp_4.x = temp_0.x - loc_0.x;
	temp_4.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_12.x * w.x - temp_12.y * w.y;
	loc_0.y = temp_12.y * w.x + temp_12.x * w.y;
	temp_12.x = temp_8.x - loc_0.x;
	temp_12.y = temp_8.y - loc_0.y;
	temp_8.x = temp_8.x + loc_0.x;
	temp_8.y = temp_8.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_6.x * iw.x - temp_6.y * iw.y;
	loc_0.y = temp_6.y * iw.x + temp_6.x * iw.y;
	temp_6.x = temp_2.x - loc_0.x;
	temp_6.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	loc_0.x = temp_14.x * iw.x - temp_14.y * iw.y;
	loc_0.y = temp_14.y * iw.x + temp_14.x * iw.y;
	temp_14.x = temp_10.x - loc_0.x;
	temp_14.y = temp_10.y - loc_0.y;
	temp_10.x = temp_10.x + loc_0.x;
	temp_10.y = temp_10.y + loc_0.y;
	iw.x = w.x * 7.07106781186547573e-01f + w.y * 7.07106781186547573e-01f;
	iw.y = w.y * 7.07106781186547573e-01f - w.x * 7.07106781186547573e-01f;

	loc_0.x = temp_5.x * iw.x - temp_5.y * iw.y;
	loc_0.y = temp_5.y * iw.x + temp_5.x * iw.y;
	temp_5.x = temp_1.x - loc_0.x;
	temp_5.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	loc_0.x = temp_13.x * iw.x - temp_13.y * iw.y;
	loc_0.y = temp_13.y * iw.x + temp_13.x * iw.y;
	temp_13.x = temp_9.x - loc_0.x;
	temp_13.y = temp_9.y - loc_0.y;
	temp_9.x = temp_9.x + loc_0.x;
	temp_9.y = temp_9.y + loc_0.y;
	w.x = iw.y;
	w.y = -iw.x;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_3.x - loc_0.x;
	temp_7.y = temp_3.y - loc_0.y;
	temp_3.x = temp_3.x + loc_0.x;
	temp_3.y = temp_3.y + loc_0.y;
	loc_0.x = temp_15.x * w.x - temp_15.y * w.y;
	loc_0.y = temp_15.y * w.x + temp_15.x * w.y;
	temp_15.x = temp_11.x - loc_0.x;
	temp_15.y = temp_11.y - loc_0.y;
	temp_11.x = temp_11.x + loc_0.x;
	temp_11.y = temp_11.y + loc_0.y;
	w.x = __cosf(0.125f*angle);
	w.y = __sinf(0.125f*angle);
	loc_0.x = temp_8.x * w.x - temp_8.y * w.y;
	loc_0.y = temp_8.y * w.x + temp_8.x * w.y;
	temp_8.x = temp_0.x - loc_0.x;
	temp_8.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_12.x * iw.x - temp_12.y * iw.y;
	loc_0.y = temp_12.y * iw.x + temp_12.x * iw.y;
	temp_12.x = temp_4.x - loc_0.x;
	temp_12.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	iw.x = w.x * 7.07106781186547573e-01f + w.y * 7.07106781186547573e-01f;
	iw.y = w.y * 7.07106781186547573e-01f - w.x * 7.07106781186547573e-01f;

	loc_0.x = temp_10.x * iw.x - temp_10.y * iw.y;
	loc_0.y = temp_10.y * iw.x + temp_10.x * iw.y;
	temp_10.x = temp_2.x - loc_0.x;
	temp_10.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	loc_0.x = iw.y;
	loc_0.y = -iw.x;
	iw = loc_0;
	loc_0.x = temp_14.x * iw.x - temp_14.y * iw.y;
	loc_0.y = temp_14.y * iw.x + temp_14.x * iw.y;
	temp_14.x = temp_6.x - loc_0.x;
	temp_14.y = temp_6.y - loc_0.y;
	temp_6.x = temp_6.x + loc_0.x;
	temp_6.y = temp_6.y + loc_0.y;
	iw.x = w.x * 9.23879532511286738e-01f + w.y * 3.82683432365089782e-01f;
	iw.y = w.y * 9.23879532511286738e-01f - w.x * 3.82683432365089782e-01f;

	loc_0.x = temp_9.x * iw.x - temp_9.y * iw.y;
	loc_0.y = temp_9.y * iw.x + temp_9.x * iw.y;
	temp_9.x = temp_1.x - loc_0.x;
	temp_9.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	loc_0.x = iw.y;
	loc_0.y = -iw.x;
	iw = loc_0;
	loc_0.x = temp_13.x * iw.x - temp_13.y * iw.y;
	loc_0.y = temp_13.y * iw.x + temp_13.x * iw.y;
	temp_13.x = temp_5.x - loc_0.x;
	temp_13.y = temp_5.y - loc_0.y;
	temp_5.x = temp_5.x + loc_0.x;
	temp_5.y = temp_5.y + loc_0.y;
	iw.x = w.x * 3.82683432365089837e-01f + w.y * 9.23879532511286738e-01f;
	iw.y = w.y * 3.82683432365089837e-01f - w.x * 9.23879532511286738e-01f;

	loc_0.x = temp_11.x * iw.x - temp_11.y * iw.y;
	loc_0.y = temp_11.y * iw.x + temp_11.x * iw.y;
	temp_11.x = temp_3.x - loc_0.x;
	temp_11.y = temp_3.y - loc_0.y;
	temp_3.x = temp_3.x + loc_0.x;
	temp_3.y = temp_3.y + loc_0.y;
	loc_0.x = iw.y;
	loc_0.y = -iw.x;
	iw = loc_0;
	loc_0.x = temp_15.x * iw.x - temp_15.y * iw.y;
	loc_0.y = temp_15.y * iw.x + temp_15.x * iw.y;
	temp_15.x = temp_7.x - loc_0.x;
	temp_15.y = temp_7.y - loc_0.y;
	temp_7.x = temp_7.x + loc_0.x;
	temp_7.y = temp_7.y + loc_0.y;
}		sharedStride = 16;
		if(disableThreads>0) {
}		if(disableThreads>0) {
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 0)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_0.x * mult.x - temp_0.y * mult.y;
		temp_0.y = temp_0.y * mult.x + temp_0.x * mult.y;
		temp_0.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 16)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_1.x * mult.x - temp_1.y * mult.y;
		temp_1.y = temp_1.y * mult.x + temp_1.x * mult.y;
		temp_1.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 32)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_2.x * mult.x - temp_2.y * mult.y;
		temp_2.y = temp_2.y * mult.x + temp_2.x * mult.y;
		temp_2.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 48)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_3.x * mult.x - temp_3.y * mult.y;
		temp_3.y = temp_3.y * mult.x + temp_3.x * mult.y;
		temp_3.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 64)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_4.x * mult.x - temp_4.y * mult.y;
		temp_4.y = temp_4.y * mult.x + temp_4.x * mult.y;
		temp_4.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 80)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_5.x * mult.x - temp_5.y * mult.y;
		temp_5.y = temp_5.y * mult.x + temp_5.x * mult.y;
		temp_5.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 96)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_6.x * mult.x - temp_6.y * mult.y;
		temp_6.y = temp_6.y * mult.x + temp_6.x * mult.y;
		temp_6.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 112)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_7.x * mult.x - temp_7.y * mult.y;
		temp_7.y = temp_7.y * mult.x + temp_7.x * mult.y;
		temp_7.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 128)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_8.x * mult.x - temp_8.y * mult.y;
		temp_8.y = temp_8.y * mult.x + temp_8.x * mult.y;
		temp_8.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 144)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_9.x * mult.x - temp_9.y * mult.y;
		temp_9.y = temp_9.y * mult.x + temp_9.x * mult.y;
		temp_9.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 160)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_10.x * mult.x - temp_10.y * mult.y;
		temp_10.y = temp_10.y * mult.x + temp_10.x * mult.y;
		temp_10.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 176)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_11.x * mult.x - temp_11.y * mult.y;
		temp_11.y = temp_11.y * mult.x + temp_11.x * mult.y;
		temp_11.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 192)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_12.x * mult.x - temp_12.y * mult.y;
		temp_12.y = temp_12.y * mult.x + temp_12.x * mult.y;
		temp_12.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 208)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_13.x * mult.x - temp_13.y * mult.y;
		temp_13.y = temp_13.y * mult.x + temp_13.x * mult.y;
		temp_13.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 224)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_14.x * mult.x - temp_14.y * mult.y;
		temp_14.y = temp_14.y * mult.x + temp_14.x * mult.y;
		temp_14.x = w.x;
		angle = 2 * 3.14159265358979312e+00f * (((((threadIdx.x + blockIdx.x * blockDim.x)) % (512)) * (threadIdx.y + 240)) / 1.31072000000000000e+05f);
		mult.x = __cosf(angle);
		mult.y = -__sinf(angle);
		w.x = temp_15.x * mult.x - temp_15.y * mult.y;
		temp_15.y = temp_15.y * mult.x + temp_15.x * mult.y;
		temp_15.x = w.x;
}		if ((((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072) < 33554432) {
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 0) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_0;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 16) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_1;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 32) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_2;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 48) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_3;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 64) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_4;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 80) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_5;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 96) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_6;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 112) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_7;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 128) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_8;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 144) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_9;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 160) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_10;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 176) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_11;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 192) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_12;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 208) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_13;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 224) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
			outputs[inoutID] = temp_14;
		inoutID = ((threadIdx.x + blockIdx.x * blockDim.x)) % (512) + 512 * (threadIdx.y + 240) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 512) * (131072);
			inoutID = (inoutID) * 1;
		if(threadIdx.y < 16){
			outputs[inoutID] = temp_15;
		}
	}
}



