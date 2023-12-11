#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>

__global__ void compute_accels(vector3** d_accels, vector3* d_hPos, double* d_mass) {
	int i, j, k;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < NUMENTITIES && j < NUMENTITIES) {
		if (i==j) {
			FILL_VECTOR(d_accels[i][j],0,0,0);
		}
		else{
			vector3 distance;
			for (k=0;k<3;k++) distance[k]=d_hPos[i][k]-d_hPos[j][k];
			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
			double magnitude=sqrt(magnitude_sq);
			double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
			FILL_VECTOR(d_accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
		}
	}
	
}

__global__ void sum_columns(vector3** d_accels, vector3* d_hVel, vector3* d_hPos) {
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	int i, j, k;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < NUMENTITIES && j < NUMENTITIES) {
		vector3 accel_sum={0,0,0};
		for (k=0;k<3;k++)
			accel_sum[k]+=d_accels[i][j][k];
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			d_hVel[i][k]+=accel_sum[k]*INTERVAL;
			d_hPos[i][k]+=d_hVel[i][k]*INTERVAL;
		}
	}
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute() {	
	dim3 threadsPerBlock(16,16);
	dim3 numBlocks(ceil((NUMENTITIES + threadsPerBlock.x-1) / threadsPerBlock.x), ceil((NUMENTITIES + threadsPerBlock.y-1) / threadsPerBlock.y));
	
	compute_accels<<<numBlocks,threadsPerBlock>>>(d_accels, d_hPos, d_mass);
	sum_columns<<<numBlocks,threadsPerBlock>>>(d_accels, d_hVel, d_hPos);
	cudaDeviceSynchronize();
}
