extern __shared__ float shared[];
extern "C" __global__ void __launch_bounds__(1024) VkFFT_main_logN13 (float2* inputs, float2* outputs) {
unsigned int sharedStride = 8192;
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
	float angle=0;
		{ 
		combinedID = threadIdx.x + 0;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		temp_0 = inputs[inoutID];
		combinedID = threadIdx.x + 1024;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		temp_1 = inputs[inoutID];
		combinedID = threadIdx.x + 2048;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		temp_2 = inputs[inoutID];
		combinedID = threadIdx.x + 3072;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		temp_3 = inputs[inoutID];
		combinedID = threadIdx.x + 4096;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		temp_4 = inputs[inoutID];
		combinedID = threadIdx.x + 5120;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		temp_5 = inputs[inoutID];
		combinedID = threadIdx.x + 6144;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		temp_6 = inputs[inoutID];
		combinedID = threadIdx.x + 7168;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		temp_7 = inputs[inoutID];
	}
		stageInvocationID = (threadIdx.x+ 0) % (1);
		angle = stageInvocationID * -3.14159265358979312e+00f;
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
	w.x = iw.y;
	w.y = -iw.x;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_6.x - loc_0.x;
	temp_7.y = temp_6.y - loc_0.y;
	temp_6.x = temp_6.x + loc_0.x;
	temp_6.y = temp_6.y + loc_0.y;
	__syncthreads();

	stageInvocationID = threadIdx.x + 0;
	blockInvocationID = stageInvocationID;
	stageInvocationID = stageInvocationID % 1;
	blockInvocationID = blockInvocationID - stageInvocationID;
	inoutID = blockInvocationID * 8;
	inoutID = inoutID + stageInvocationID;
	sdataID = inoutID + 0;
	sdata[sdataID] = temp_0;
	sdataID = inoutID + 1;
	sdata[sdataID] = temp_4;
	sdataID = inoutID + 2;
	sdata[sdataID] = temp_2;
	sdataID = inoutID + 3;
	sdata[sdataID] = temp_6;
	sdataID = inoutID + 4;
	sdata[sdataID] = temp_1;
	sdataID = inoutID + 5;
	sdata[sdataID] = temp_5;
	sdataID = inoutID + 6;
	sdata[sdataID] = temp_3;
	sdataID = inoutID + 7;
	sdata[sdataID] = temp_7;
	__syncthreads();

		stageInvocationID = (threadIdx.x+ 0) % (8);
		angle = stageInvocationID * -3.92699081698724139e-01f;
		sdataID = threadIdx.x + 0;
		temp_0 = sdata[sdataID];
		sdataID = threadIdx.x + 1024;
		temp_4 = sdata[sdataID];
		sdataID = threadIdx.x + 2048;
		temp_2 = sdata[sdataID];
		sdataID = threadIdx.x + 3072;
		temp_6 = sdata[sdataID];
		sdataID = threadIdx.x + 4096;
		temp_1 = sdata[sdataID];
		sdataID = threadIdx.x + 5120;
		temp_5 = sdata[sdataID];
		sdataID = threadIdx.x + 6144;
		temp_3 = sdata[sdataID];
		sdataID = threadIdx.x + 7168;
		temp_7 = sdata[sdataID];
	w.x = __cosf(angle);
	w.y = __sinf(angle);
	loc_0.x = temp_1.x * w.x - temp_1.y * w.y;
	loc_0.y = temp_1.y * w.x + temp_1.x * w.y;
	temp_1.x = temp_0.x - loc_0.x;
	temp_1.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_5.x * w.x - temp_5.y * w.y;
	loc_0.y = temp_5.y * w.x + temp_5.x * w.y;
	temp_5.x = temp_4.x - loc_0.x;
	temp_5.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	loc_0.x = temp_3.x * w.x - temp_3.y * w.y;
	loc_0.y = temp_3.y * w.x + temp_3.x * w.y;
	temp_3.x = temp_2.x - loc_0.x;
	temp_3.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_6.x - loc_0.x;
	temp_7.y = temp_6.y - loc_0.y;
	temp_6.x = temp_6.x + loc_0.x;
	temp_6.y = temp_6.y + loc_0.y;
	w.x = __cosf(0.5f*angle);
	w.y = __sinf(0.5f*angle);
	loc_0.x = temp_2.x * w.x - temp_2.y * w.y;
	loc_0.y = temp_2.y * w.x + temp_2.x * w.y;
	temp_2.x = temp_0.x - loc_0.x;
	temp_2.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_6.x * w.x - temp_6.y * w.y;
	loc_0.y = temp_6.y * w.x + temp_6.x * w.y;
	temp_6.x = temp_4.x - loc_0.x;
	temp_6.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_3.x * iw.x - temp_3.y * iw.y;
	loc_0.y = temp_3.y * iw.x + temp_3.x * iw.y;
	temp_3.x = temp_1.x - loc_0.x;
	temp_3.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	loc_0.x = temp_7.x * iw.x - temp_7.y * iw.y;
	loc_0.y = temp_7.y * iw.x + temp_7.x * iw.y;
	temp_7.x = temp_5.x - loc_0.x;
	temp_7.y = temp_5.y - loc_0.y;
	temp_5.x = temp_5.x + loc_0.x;
	temp_5.y = temp_5.y + loc_0.y;
	w.x = __cosf(0.25f*angle);
	w.y = __sinf(0.25f*angle);
	loc_0.x = temp_4.x * w.x - temp_4.y * w.y;
	loc_0.y = temp_4.y * w.x + temp_4.x * w.y;
	temp_4.x = temp_0.x - loc_0.x;
	temp_4.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_6.x * iw.x - temp_6.y * iw.y;
	loc_0.y = temp_6.y * iw.x + temp_6.x * iw.y;
	temp_6.x = temp_2.x - loc_0.x;
	temp_6.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	iw.x = w.x * 7.07106781186547573e-01f + w.y * 7.07106781186547573e-01f;
	iw.y = w.y * 7.07106781186547573e-01f - w.x * 7.07106781186547573e-01f;

	loc_0.x = temp_5.x * iw.x - temp_5.y * iw.y;
	loc_0.y = temp_5.y * iw.x + temp_5.x * iw.y;
	temp_5.x = temp_1.x - loc_0.x;
	temp_5.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	w.x = iw.y;
	w.y = -iw.x;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_3.x - loc_0.x;
	temp_7.y = temp_3.y - loc_0.y;
	temp_3.x = temp_3.x + loc_0.x;
	temp_3.y = temp_3.y + loc_0.y;
	__syncthreads();

	stageInvocationID = threadIdx.x + 0;
	blockInvocationID = stageInvocationID;
	stageInvocationID = stageInvocationID % 8;
	blockInvocationID = blockInvocationID - stageInvocationID;
	inoutID = blockInvocationID * 8;
	inoutID = inoutID + stageInvocationID;
	sdataID = inoutID + 0;
	sdata[sdataID] = temp_0;
	sdataID = inoutID + 8;
	sdata[sdataID] = temp_1;
	sdataID = inoutID + 16;
	sdata[sdataID] = temp_2;
	sdataID = inoutID + 24;
	sdata[sdataID] = temp_3;
	sdataID = inoutID + 32;
	sdata[sdataID] = temp_4;
	sdataID = inoutID + 40;
	sdata[sdataID] = temp_5;
	sdataID = inoutID + 48;
	sdata[sdataID] = temp_6;
	sdataID = inoutID + 56;
	sdata[sdataID] = temp_7;
	__syncthreads();

		stageInvocationID = (threadIdx.x+ 0) % (64);
		angle = stageInvocationID * -4.90873852123405174e-02f;
		sdataID = threadIdx.x + 0;
		temp_0 = sdata[sdataID];
		sdataID = threadIdx.x + 1024;
		temp_1 = sdata[sdataID];
		sdataID = threadIdx.x + 2048;
		temp_2 = sdata[sdataID];
		sdataID = threadIdx.x + 3072;
		temp_3 = sdata[sdataID];
		sdataID = threadIdx.x + 4096;
		temp_4 = sdata[sdataID];
		sdataID = threadIdx.x + 5120;
		temp_5 = sdata[sdataID];
		sdataID = threadIdx.x + 6144;
		temp_6 = sdata[sdataID];
		sdataID = threadIdx.x + 7168;
		temp_7 = sdata[sdataID];
	w.x = __cosf(angle);
	w.y = __sinf(angle);
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
	w.x = __cosf(0.5f*angle);
	w.y = __sinf(0.5f*angle);
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
	w.x = __cosf(0.25f*angle);
	w.y = __sinf(0.25f*angle);
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
	w.x = iw.y;
	w.y = -iw.x;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_6.x - loc_0.x;
	temp_7.y = temp_6.y - loc_0.y;
	temp_6.x = temp_6.x + loc_0.x;
	temp_6.y = temp_6.y + loc_0.y;
	__syncthreads();

	stageInvocationID = threadIdx.x + 0;
	blockInvocationID = stageInvocationID;
	stageInvocationID = stageInvocationID % 64;
	blockInvocationID = blockInvocationID - stageInvocationID;
	inoutID = blockInvocationID * 8;
	inoutID = inoutID + stageInvocationID;
	sdataID = inoutID + 0;
	sdata[sdataID] = temp_0;
	sdataID = inoutID + 64;
	sdata[sdataID] = temp_4;
	sdataID = inoutID + 128;
	sdata[sdataID] = temp_2;
	sdataID = inoutID + 192;
	sdata[sdataID] = temp_6;
	sdataID = inoutID + 256;
	sdata[sdataID] = temp_1;
	sdataID = inoutID + 320;
	sdata[sdataID] = temp_5;
	sdataID = inoutID + 384;
	sdata[sdataID] = temp_3;
	sdataID = inoutID + 448;
	sdata[sdataID] = temp_7;
	__syncthreads();

		stageInvocationID = (threadIdx.x+ 0) % (512);
		angle = stageInvocationID * -6.13592315154256468e-03f;
		sdataID = threadIdx.x + 0;
		temp_0 = sdata[sdataID];
		sdataID = threadIdx.x + 1024;
		temp_4 = sdata[sdataID];
		sdataID = threadIdx.x + 2048;
		temp_2 = sdata[sdataID];
		sdataID = threadIdx.x + 3072;
		temp_6 = sdata[sdataID];
		sdataID = threadIdx.x + 4096;
		temp_1 = sdata[sdataID];
		sdataID = threadIdx.x + 5120;
		temp_5 = sdata[sdataID];
		sdataID = threadIdx.x + 6144;
		temp_3 = sdata[sdataID];
		sdataID = threadIdx.x + 7168;
		temp_7 = sdata[sdataID];
	w.x = __cosf(angle);
	w.y = __sinf(angle);
	loc_0.x = temp_1.x * w.x - temp_1.y * w.y;
	loc_0.y = temp_1.y * w.x + temp_1.x * w.y;
	temp_1.x = temp_0.x - loc_0.x;
	temp_1.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_5.x * w.x - temp_5.y * w.y;
	loc_0.y = temp_5.y * w.x + temp_5.x * w.y;
	temp_5.x = temp_4.x - loc_0.x;
	temp_5.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	loc_0.x = temp_3.x * w.x - temp_3.y * w.y;
	loc_0.y = temp_3.y * w.x + temp_3.x * w.y;
	temp_3.x = temp_2.x - loc_0.x;
	temp_3.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_6.x - loc_0.x;
	temp_7.y = temp_6.y - loc_0.y;
	temp_6.x = temp_6.x + loc_0.x;
	temp_6.y = temp_6.y + loc_0.y;
	w.x = __cosf(0.5f*angle);
	w.y = __sinf(0.5f*angle);
	loc_0.x = temp_2.x * w.x - temp_2.y * w.y;
	loc_0.y = temp_2.y * w.x + temp_2.x * w.y;
	temp_2.x = temp_0.x - loc_0.x;
	temp_2.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_6.x * w.x - temp_6.y * w.y;
	loc_0.y = temp_6.y * w.x + temp_6.x * w.y;
	temp_6.x = temp_4.x - loc_0.x;
	temp_6.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_3.x * iw.x - temp_3.y * iw.y;
	loc_0.y = temp_3.y * iw.x + temp_3.x * iw.y;
	temp_3.x = temp_1.x - loc_0.x;
	temp_3.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	loc_0.x = temp_7.x * iw.x - temp_7.y * iw.y;
	loc_0.y = temp_7.y * iw.x + temp_7.x * iw.y;
	temp_7.x = temp_5.x - loc_0.x;
	temp_7.y = temp_5.y - loc_0.y;
	temp_5.x = temp_5.x + loc_0.x;
	temp_5.y = temp_5.y + loc_0.y;
	w.x = __cosf(0.25f*angle);
	w.y = __sinf(0.25f*angle);
	loc_0.x = temp_4.x * w.x - temp_4.y * w.y;
	loc_0.y = temp_4.y * w.x + temp_4.x * w.y;
	temp_4.x = temp_0.x - loc_0.x;
	temp_4.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_6.x * iw.x - temp_6.y * iw.y;
	loc_0.y = temp_6.y * iw.x + temp_6.x * iw.y;
	temp_6.x = temp_2.x - loc_0.x;
	temp_6.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	iw.x = w.x * 7.07106781186547573e-01f + w.y * 7.07106781186547573e-01f;
	iw.y = w.y * 7.07106781186547573e-01f - w.x * 7.07106781186547573e-01f;

	loc_0.x = temp_5.x * iw.x - temp_5.y * iw.y;
	loc_0.y = temp_5.y * iw.x + temp_5.x * iw.y;
	temp_5.x = temp_1.x - loc_0.x;
	temp_5.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	w.x = iw.y;
	w.y = -iw.x;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_3.x - loc_0.x;
	temp_7.y = temp_3.y - loc_0.y;
	temp_3.x = temp_3.x + loc_0.x;
	temp_3.y = temp_3.y + loc_0.y;
	__syncthreads();

	stageInvocationID = threadIdx.x + 0;
	blockInvocationID = stageInvocationID;
	stageInvocationID = stageInvocationID % 512;
	blockInvocationID = blockInvocationID - stageInvocationID;
	inoutID = blockInvocationID * 8;
	inoutID = inoutID + stageInvocationID;
	sdataID = inoutID + 0;
	sdata[sdataID] = temp_0;
	sdataID = inoutID + 512;
	sdata[sdataID] = temp_1;
	sdataID = inoutID + 1024;
	sdata[sdataID] = temp_2;
	sdataID = inoutID + 1536;
	sdata[sdataID] = temp_3;
	sdataID = inoutID + 2048;
	sdata[sdataID] = temp_4;
	sdataID = inoutID + 2560;
	sdata[sdataID] = temp_5;
	sdataID = inoutID + 3072;
	sdata[sdataID] = temp_6;
	sdataID = inoutID + 3584;
	sdata[sdataID] = temp_7;
	__syncthreads();

		stageInvocationID = (threadIdx.x+ 0) % (4096);
		angle = stageInvocationID * -7.66990393942820585e-04f;
		sdataID = threadIdx.x + 0;
		temp_0 = sdata[sdataID];
		sdataID = threadIdx.x + 4096;
		temp_4 = sdata[sdataID];
	w.x = __cosf(angle);
	w.y = __sinf(angle);
	loc_0.x = temp_4.x * w.x - temp_4.y * w.y;
	loc_0.y = temp_4.y * w.x + temp_4.x * w.y;
	temp_4.x = temp_0.x - loc_0.x;
	temp_4.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
		stageInvocationID = (threadIdx.x+ 1024) % (4096);
		angle = stageInvocationID * -7.66990393942820585e-04f;
		sdataID = threadIdx.x + 1024;
		temp_1 = sdata[sdataID];
		sdataID = threadIdx.x + 5120;
		temp_5 = sdata[sdataID];
	w.x = __cosf(angle);
	w.y = __sinf(angle);
	loc_0.x = temp_5.x * w.x - temp_5.y * w.y;
	loc_0.y = temp_5.y * w.x + temp_5.x * w.y;
	temp_5.x = temp_1.x - loc_0.x;
	temp_5.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
		stageInvocationID = (threadIdx.x+ 2048) % (4096);
		angle = stageInvocationID * -7.66990393942820585e-04f;
		sdataID = threadIdx.x + 2048;
		temp_2 = sdata[sdataID];
		sdataID = threadIdx.x + 6144;
		temp_6 = sdata[sdataID];
	w.x = __cosf(angle);
	w.y = __sinf(angle);
	loc_0.x = temp_6.x * w.x - temp_6.y * w.y;
	loc_0.y = temp_6.y * w.x + temp_6.x * w.y;
	temp_6.x = temp_2.x - loc_0.x;
	temp_6.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
		stageInvocationID = (threadIdx.x+ 3072) % (4096);
		angle = stageInvocationID * -7.66990393942820585e-04f;
		sdataID = threadIdx.x + 3072;
		temp_3 = sdata[sdataID];
		sdataID = threadIdx.x + 7168;
		temp_7 = sdata[sdataID];
	w.x = __cosf(angle);
	w.y = __sinf(angle);
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_3.x - loc_0.x;
	temp_7.y = temp_3.y - loc_0.y;
	temp_3.x = temp_3.x + loc_0.x;
	temp_3.y = temp_3.y + loc_0.y;
		{ 
		combinedID = threadIdx.x + 0;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		outputs[inoutID] = temp_0;
		combinedID = threadIdx.x + 1024;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		outputs[inoutID] = temp_1;
		combinedID = threadIdx.x + 2048;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		outputs[inoutID] = temp_2;
		combinedID = threadIdx.x + 3072;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		outputs[inoutID] = temp_3;
		combinedID = threadIdx.x + 4096;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		outputs[inoutID] = temp_4;
		combinedID = threadIdx.x + 5120;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		outputs[inoutID] = temp_5;
		combinedID = threadIdx.x + 6144;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		outputs[inoutID] = temp_6;
		combinedID = threadIdx.x + 7168;
		inoutID = (combinedID % 8192) + (combinedID / 8192) * 8192;
			inoutID = (inoutID);
		outputs[inoutID] = temp_7;
	}
}



