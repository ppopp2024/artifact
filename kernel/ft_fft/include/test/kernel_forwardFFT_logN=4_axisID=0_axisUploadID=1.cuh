extern __shared__ float shared[];
extern "C" __global__ void __launch_bounds__(4) VkFFT_main_logN4 (float2* inputs, float2* outputs) {
unsigned int sharedStride = 32;
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
	float2 w;
	w.x=0;
	w.y=0;
	float2 loc_0;
	loc_0.x=0;
	loc_0.y=0;
	unsigned int stageInvocationID=0;
	unsigned int blockInvocationID=0;
	unsigned int sdataID=0;
	unsigned int combinedID=0;
	unsigned int inoutID=0;
	float angle=0;
		{ 
		combinedID = threadIdx.x + 0;
		inoutID = (combinedID % 16) + (combinedID / 16) * 16;
			inoutID = (inoutID);
		temp_0 = inputs[inoutID];
		combinedID = threadIdx.x + 4;
		inoutID = (combinedID % 16) + (combinedID / 16) * 16;
			inoutID = (inoutID);
		temp_1 = inputs[inoutID];
		combinedID = threadIdx.x + 8;
		inoutID = (combinedID % 16) + (combinedID / 16) * 16;
			inoutID = (inoutID);
		temp_2 = inputs[inoutID];
		combinedID = threadIdx.x + 12;
		inoutID = (combinedID % 16) + (combinedID / 16) * 16;
			inoutID = (inoutID);
		temp_3 = inputs[inoutID];
	}
		stageInvocationID = (threadIdx.x+ 0) % (1);
		angle = stageInvocationID * -3.14159265358979312e+00f;
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
	w.x = 1;
	w.y = 0;
	loc_0.x = temp_1.x * w.x - temp_1.y * w.y;
	loc_0.y = temp_1.y * w.x + temp_1.x * w.y;
	temp_1.x = temp_0.x - loc_0.x;
	temp_1.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = w.x;	w.x = w.y;
	w.y = -loc_0.x;
	loc_0.x = temp_3.x * w.x - temp_3.y * w.y;
	loc_0.y = temp_3.y * w.x + temp_3.x * w.y;
	temp_3.x = temp_2.x - loc_0.x;
	temp_3.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	__syncthreads();

	stageInvocationID = threadIdx.x + 0;
	blockInvocationID = stageInvocationID;
	stageInvocationID = stageInvocationID % 1;
	blockInvocationID = blockInvocationID - stageInvocationID;
	inoutID = blockInvocationID * 4;
	inoutID = inoutID + stageInvocationID;
	sdataID = inoutID + 0;
	sdata[sdataID] = temp_0;
	sdataID = inoutID + 1;
	sdata[sdataID] = temp_2;
	sdataID = inoutID + 2;
	sdata[sdataID] = temp_1;
	sdataID = inoutID + 3;
	sdata[sdataID] = temp_3;
	__syncthreads();

		stageInvocationID = (threadIdx.x+ 0) % (4);
		angle = stageInvocationID * -7.85398163397448279e-01f;
		sdataID = threadIdx.x + 0;
		temp_0 = sdata[sdataID];
		sdataID = threadIdx.x + 4;
		temp_2 = sdata[sdataID];
		sdataID = threadIdx.x + 8;
		temp_1 = sdata[sdataID];
		sdataID = threadIdx.x + 12;
		temp_3 = sdata[sdataID];
	w.x = __cosf(angle);
	w.y = __sinf(angle);
	loc_0.x = temp_1.x * w.x - temp_1.y * w.y;
	loc_0.y = temp_1.y * w.x + temp_1.x * w.y;
	temp_1.x = temp_0.x - loc_0.x;
	temp_1.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_3.x * w.x - temp_3.y * w.y;
	loc_0.y = temp_3.y * w.x + temp_3.x * w.y;
	temp_3.x = temp_2.x - loc_0.x;
	temp_3.y = temp_2.y - loc_0.y;
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
	loc_0.x = temp_3.x * w.x - temp_3.y * w.y;
	loc_0.y = temp_3.y * w.x + temp_3.x * w.y;
	temp_3.x = temp_1.x - loc_0.x;
	temp_3.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
		{ 
		combinedID = threadIdx.x + 0;
		inoutID = (combinedID % 16) + (combinedID / 16) * 16;
			inoutID = (inoutID);
		outputs[inoutID] = temp_0;
		combinedID = threadIdx.x + 4;
		inoutID = (combinedID % 16) + (combinedID / 16) * 16;
			inoutID = (inoutID);
		outputs[inoutID] = temp_1;
		combinedID = threadIdx.x + 8;
		inoutID = (combinedID % 16) + (combinedID / 16) * 16;
			inoutID = (inoutID);
		outputs[inoutID] = temp_2;
		combinedID = threadIdx.x + 12;
		inoutID = (combinedID % 16) + (combinedID / 16) * 16;
			inoutID = (inoutID);
		outputs[inoutID] = temp_3;
	}
}


