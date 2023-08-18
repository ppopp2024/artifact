extern __shared__ float shared[];
extern "C" __global__ void __launch_bounds__(32) VkFFT_main_logN8 (float2* inputs, float2* outputs) {
unsigned int sharedStride = 272;
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
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		temp_0 = inputs[inoutID];
		combinedID = threadIdx.x + 32;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		temp_1 = inputs[inoutID];
		combinedID = threadIdx.x + 64;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		temp_2 = inputs[inoutID];
		combinedID = threadIdx.x + 96;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		temp_3 = inputs[inoutID];
		combinedID = threadIdx.x + 128;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		temp_4 = inputs[inoutID];
		combinedID = threadIdx.x + 160;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		temp_5 = inputs[inoutID];
		combinedID = threadIdx.x + 192;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		temp_6 = inputs[inoutID];
		combinedID = threadIdx.x + 224;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
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
	sharedStride = 272;	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_0;
	sdataID = inoutID + 1;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_4;
	sdataID = inoutID + 2;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_2;
	sdataID = inoutID + 3;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_6;
	sdataID = inoutID + 4;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_1;
	sdataID = inoutID + 5;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_5;
	sdataID = inoutID + 6;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_3;
	sdataID = inoutID + 7;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_7;
	__syncthreads();

		stageInvocationID = (threadIdx.x+ 0) % (8);
		angle = stageInvocationID * -3.92699081698724139e-01f;
		sdataID = threadIdx.x + 0;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_0 = sdata[sdataID];
		sdataID = threadIdx.x + 32;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_4 = sdata[sdataID];
		sdataID = threadIdx.x + 64;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_2 = sdata[sdataID];
		sdataID = threadIdx.x + 96;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_6 = sdata[sdataID];
		sdataID = threadIdx.x + 128;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_1 = sdata[sdataID];
		sdataID = threadIdx.x + 160;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_5 = sdata[sdataID];
		sdataID = threadIdx.x + 192;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_3 = sdata[sdataID];
		sdataID = threadIdx.x + 224;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_7 = sdata[sdataID];
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
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_0;
	sdataID = inoutID + 8;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_1;
	sdataID = inoutID + 16;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_2;
	sdataID = inoutID + 24;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_3;
	sdataID = inoutID + 32;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_4;
	sdataID = inoutID + 40;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_5;
	sdataID = inoutID + 48;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_6;
	sdataID = inoutID + 56;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;	sdata[sdataID] = temp_7;
	__syncthreads();

		stageInvocationID = (threadIdx.x+ 0) % (64);
		angle = stageInvocationID * -4.90873852123405174e-02f;
		sdataID = threadIdx.x + 0;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_0 = sdata[sdataID];
		sdataID = threadIdx.x + 64;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_2 = sdata[sdataID];
		sdataID = threadIdx.x + 128;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_4 = sdata[sdataID];
		sdataID = threadIdx.x + 192;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_6 = sdata[sdataID];
	stageInvocationID = (threadIdx.x+ 32) % (64);
		// angle = stageInvocationID * -4.90873852123405174e-02f;
		sdataID = threadIdx.x + 32;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_1 = sdata[sdataID];
		sdataID = threadIdx.x + 96;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_3 = sdata[sdataID];
		sdataID = threadIdx.x + 160;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_5 = sdata[sdataID];
		sdataID = threadIdx.x + 224;
	sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_7 = sdata[sdataID];
	w.x = __cosf(angle);
	w.y = __sinf(angle);
	loc_0.x = temp_4.x * w.x - temp_4.y * w.y;
	loc_0.y = temp_4.y * w.x + temp_4.x * w.y;
	temp_4.x = temp_0.x - loc_0.x;
	temp_4.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_6.x * w.x - temp_6.y * w.y;
	loc_0.y = temp_6.y * w.x + temp_6.x * w.y;
	temp_6.x = temp_2.x - loc_0.x;
	temp_6.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	w.x = __cosf(0.5f*angle);
	w.y = __sinf(0.5f*angle);
	loc_0.x = temp_2.x * w.x - temp_2.y * w.y;
	loc_0.y = temp_2.y * w.x + temp_2.x * w.y;
	temp_2.x = temp_0.x - loc_0.x;
	temp_2.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = w.x;	w.x = w.y;
	w.y = -loc_0.x;
	loc_0.x = temp_6.x * w.x - temp_6.y * w.y;
	loc_0.y = temp_6.y * w.x + temp_6.x * w.y;
	temp_6.x = temp_4.x - loc_0.x;
	temp_6.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	// 	stageInvocationID = (threadIdx.x+ 32) % (64);
		angle = stageInvocationID * -4.90873852123405174e-02f;
	// 	sdataID = threadIdx.x + 32;
	// sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_1 = sdata[sdataID];
	// 	sdataID = threadIdx.x + 96;
	// sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_3 = sdata[sdataID];
	// 	sdataID = threadIdx.x + 160;
	// sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_5 = sdata[sdataID];
	// 	sdataID = threadIdx.x + 224;
	// sdataID = (sdataID / 16) * 17 + sdataID % 16;		temp_7 = sdata[sdataID];
	w.x = __cosf(angle);
	w.y = __sinf(angle);
	loc_0.x = temp_5.x * w.x - temp_5.y * w.y;
	loc_0.y = temp_5.y * w.x + temp_5.x * w.y;
	temp_5.x = temp_1.x - loc_0.x;
	temp_5.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_3.x - loc_0.x;
	temp_7.y = temp_3.y - loc_0.y;
	temp_3.x = temp_3.x + loc_0.x;
	temp_3.y = temp_3.y + loc_0.y;
	w.x = __cosf(0.5f*angle);
	w.y = __sinf(0.5f*angle);
	loc_0.x = temp_3.x * w.x - temp_3.y * w.y;
	loc_0.y = temp_3.y * w.x + temp_3.x * w.y;
	temp_3.x = temp_1.x - loc_0.x;
	temp_3.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	loc_0.x = w.x;	w.x = w.y;
	w.y = -loc_0.x;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_5.x - loc_0.x;
	temp_7.y = temp_5.y - loc_0.y;
	temp_5.x = temp_5.x + loc_0.x;
	temp_5.y = temp_5.y + loc_0.y;
		{ 
		combinedID = threadIdx.x + 0;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		outputs[inoutID] = temp_0;
		combinedID = threadIdx.x + 32;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		outputs[inoutID] = temp_1;
		combinedID = threadIdx.x + 64;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		outputs[inoutID] = temp_4;
		combinedID = threadIdx.x + 96;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		outputs[inoutID] = temp_5;
		combinedID = threadIdx.x + 128;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		outputs[inoutID] = temp_2;
		combinedID = threadIdx.x + 160;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		outputs[inoutID] = temp_3;
		combinedID = threadIdx.x + 192;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		outputs[inoutID] = temp_6;
		combinedID = threadIdx.x + 224;
		inoutID = (combinedID % 256) + (combinedID / 256) * 256;
			inoutID = (inoutID);
		outputs[inoutID] = temp_7;
	}
}



