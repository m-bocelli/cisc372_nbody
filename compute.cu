#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>

__global__ void allocate_accels(vector3** d_accels, vector3* d_values) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NUMENTITIES)
		d_accels[i]=&d_values[i*NUMENTITIES];
}

__global__ void compute_accels(vector3** d_accels, vector3* d_hVel, vector3* d_hPos, double* d_mass) {
	int i,j,k;
	for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
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
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=d_accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			d_hVel[i][k]+=accel_sum[k]*INTERVAL;
			d_hPos[i][k]+=d_hVel[i][k]*INTERVAL;
		}
	}
}

__global__ void print_accels(vector3** d_accels) {
	for (int i = 0; i < NUMENTITIES; i++) {
		for (int j = 0; j < NUMENTITIES; j++) {
			printf("%d   ", d_accels[i][j][0]);
		}
		printf("\n");
	}
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	vector3* d_values;
	vector3** d_accels;
	cudaMalloc(&d_values, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	cudaMalloc(&d_accels, sizeof(vector3*)*NUMENTITIES);
	dim3 threadsPerBlock(16,16);
	int numBlocks = (NUMENTITIES + threadsPerBlock.x - 1) / threadsPerBlock.x;

	allocate_accels<<<numBlocks,threadsPerBlock>>>(d_accels, d_values);
	//compute_accels<<<1, 1>>>(d_accels, d_hVel, d_hPos, d_mass);
	//print_accels<<<1, 1>>>(d_accels);
	cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	/*
	//first compute the pairwise accelerations.  Effect is on the first argument.
	for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}

	/* for reduction
	for (i=0;i<NUMENTITIES;i++) {
		for (k=0;k<3;k++){ 
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}
	*/
	cudaFree(d_values);
	cudaFree(d_accels);
}
