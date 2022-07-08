//*************************************************************************************************************
//                                  Programa CUDA/C de mosquitas para N manzanas version (2021) 
//                                  Autoras Ana A. Gramajo y Karina Laneri
//*************************************************************************************************************
/*
The simulation setup:
	- Spatiality. We consider a square grid of size L x L = M, the number of blocks of the simulated city. 

	- Initial conditions.  We define the total number of mosquitoes in the system N(t=0), i.e., initially we set:
		State[i]: 0 (alive).
		Age[i]: a random number between 19 and 25 days (extracted from a uniform distribution).
    		Bucket[i]: i. 
    		Block[i]: a random number between 0 and M (extracted from a uniform distribution).
    		Adulthood[i]: the pupa matures between 17 and 19 days (extracted from a uniform distribution).
    		LifeSpan[i]: a random number between 27 and 32 days (extracted from a uniform distribution).
We choose the buckets randomly distributed on the blocks with up to 10 buckets per block. In addition, we initially consider one mosquito per bucket. 
Then, we calculate the daily mosquito population along a year. For simplicity, we consider 400 days for the simulation. 

Every day we computed the adult and aquatic population considering:

	- Oviposition. According to the model presented by Otero2006,  the average number of eggs laid by one adult female in one oviposition is 63, while the field data obtained by Bergero \textit{et al.} in Ref. \cite{Bergero2012} gives a number of eggs much lower. % than those proposed by Otero \textit{et al}. 
	Following the Bergero2012 results, we assume that the female lays between 10 and 35 eggs per oviposition (extracted from a uniform distribution). 
	Moreover, for the number of ovipositions that each female can have, we consider the model given by Otero2006: 
		- one at 18°C,
		- four or five at 23°C and 27°C, 
		- six at 30°C.

	- Mortality rates. We consider the daily mortality of eggs, pupae, larvae, and adults independent of temperature and density. The data was extracted from Otero2006 where:
        	- The mortality of the eggs is chosen to be 0.01 1/day.
        	- The death of the larvae is approximated by 0.01 1/day.
        	- The intrinsic mortality of a pupa has been considered as 0.01 1/day.
        	- The daily mortality in the pupal stage associated with the unsuccessful emergence of the adult individual is assume a mortality of 0.171/day.
We consider that the mortality of adults is 0.01 1/day according to the number of days that each individual lives.

	- Death by old age. The mosquito dies when it reaches its chosen life span.

	- Seasons. We discretize the data corresponding to the mean daily temperatures for the period July 2001–July 2002 used by Otero2008 from Buenos Aires, Argentina. The simulation starts on July 1st and, the temperature at different seasons is defined as follows:
    		-  T=18°C in the range of days [1,80) and (320,400]. 
    		-  T=23°C between [80, 140] days. 
    		-  T=30°C within (140,260) days.
    		-  T=27°C between  [260,320] days. 

	- Saturation of the buckets. We suppose a maximum number of eggs allowed per bucket. This limit is calculated from the knowledge that females lay eggs on wet surfaces just above the water level of the container. Therefore, we assume that for a  diameter container of d = 20 cm with a circumference length given by L = \pi x d = 630 mm and an egg size of s = 0.8 mm, the number of eggs covering the entire perimeter of the garbage is L/s=800. 

	- Bucket transfer. In the present model, when a bucket is saturated by mosquitoes in aquatic stage, the female move to another bucket for oviposition. Previous works (boyce1911,reiter1995,reiter1996) estimated that the range of flight of Ae. ae has a minimum of 100 m in an urban area.
As result, we assume that adult females can move with a higher probability to another bucket into the same block (80 %) and only can move with a lower probability to the nearest neighbor (20 %).

	- Advertising campaigns. We introduce the discarding effectiveness through the emptying of a percentage (PROP) of the buckets per block during the hottest months, i.e., December, January, and February. In our model, this season corresponds to the range of days (120,320). 

	- Frequency of discarding (TIEMPODESCACH). We chose from a uniform distribution in the interval [1,14] for each container (unless otherwise indicated). In this way, all the buckets are not synchronized to emptying at once.

	- Time-delay in the bucket disponibility (nTau) Once the buckets are emptied during the advertising campaigns, we add a time-delay measured in days until being available.

	- Aquatics hibernation. We include a delay on the development of eggs, larval and pupal stages in the coolest days, i.e., between [1,80) and (320,400) days. 

 	- Daily count. We count daily the number of adult and aquatic individuals.

	- Daily mosquito population aging. We increase the age of the mosquitoes by one day.   
---------------------------------------------------------------------
*/

#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG


#include <random>

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include "ran2.h"
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include "gpu_timer.h"
#include "parametros.h"

#include<stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



using namespace thrust::placeholders;

FILE *archivoT=NULL, *archivoAd=NULL, *archivoAc=NULL, *archivo1=NULL, *archivo2=NULL, *archivo3=NULL;
char miarch[50];

int tiempo_entre_oviposiciones(int dia){
	int t;
	//defino tiempos de oviposición y maduración en función de la temperatura	
	if(dia >= 1 && dia < 80) t=TOVIP3;      //<T>=18 
	if(dia >= 80 && dia <= 140)t=TOVIP2b;   //<T>=23
  	if(dia > 140 || dia < 260) t=TOVIP1;    //<T>=30
 	if(dia >= 260 && dia <= 320)t=TOVIP2a;  //<T>=27
 	if(dia > 320) t=TOVIP3;                 //<T>=18

	return t;}

__global__ void kernel_reproducir(int *estado, int *edad, int *tacho,int *TdV, int *pupacion, int *manzana, int *N_mobil, int dia, int tovip, int *nacidos)
{                                   

    	int indice=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;
		/*Si la mosquita esta viva, esta en edad adulta, en tiempo de oviposicion y vive en un tacho disponible entonces*/
  		if(id < indice && edad[id] > pupacion[id] && edad[id]%tovip == 0) 

  		{
			RNG philox;         
			RNG::ctr_type c={{}};
			RNG::key_type k={{}};
			RNG::ctr_type r;
			k[0]=id; 
			c[1]=dia;
			c[0]=SEMILLAGLOBAL; 
			
			r = philox(c, k); 
			double azar=(u01_closed_open_32_53(r[0]));//numero aleatorios entre [0,1)
        
			// estado[id] == ESTADOVIVO && edad[id]%tovip == 0){
        
			/*Si el tacho en el que nacio tiene lugar entonces*/
				/*Antes estaba asi y andaba...*/
			int tach=tacho[id];     //tach es un entero que me indica el numero de tacho en el que esta cada mosquita
				

			int iovip=10+ (azar*25); //iovip es el numero de huevos que pone cada mosquita
						
			atomicAdd(nacidos+tach,iovip); /*sumo iovip HUEVOS en la posicion del vector nacidos (que tiene NTACHOS elementos)
				nacidos[0+tach] en el puntero al primer elemento del vector nacidos desplazado en tach elementos
				el vector nacidos tiene el numero de nacidos en cada tacho Ej: nacidos[0]=numero de nacidos en el tacho 1*/
				

  		}//cierro loop de hilos de mosquitas VIVAS   
};

//mortalidades varias	
__global__ void matar_kernel(int *estado, int *edad, int *tacho,int *pupacion, int *TdV,int *N_mobil, int dia)
{
	int N=N_mobil[0];
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id<N){ 	
	 	RNG philox;         
	    RNG::ctr_type c={{}};
	    RNG::key_type k={{}};
	    RNG::ctr_type r;
	    k[0]=id; 
	    c[1]=dia;
	    c[0]=SEMILLAGLOBAL; 
		
    	r = philox(c, k); 
    	double azar;

     	azar=(u01_closed_open_32_53(r[0]));//numero aleatorios entre [0,1)
     	if (edad[id] < pupacion[id]){if(azar < MORACU)estado[id]=ESTADOMUERTO;}

     	azar=(u01_closed_open_32_53(r[0]));
     	if (edad[id] == pupacion[id]){if(azar < MORPUPAD)estado[id]=ESTADOMUERTO;}

     	azar=(u01_closed_open_32_53(r[0]));
		if (edad[id] > pupacion[id]){if(azar < MORAD)estado[id]=ESTADOMUERTO;}     	

        //matar viejos
		if (edad[id] >= TdV[id]){estado[id]=ESTADOMUERTO;}
	}
};

__global__ void descacharrado_kernel(int *estado, int *edad, int *tacho, int *pupacion,int *N_mobil,int dia, bool *mask)
{
	
    	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

	    if(id<N){// esta condicion es igual a decir que la mosquita estรก viva
	        if (edad[id] < pupacion[id] && mask[tacho[id]]==1){ //NUEVOKARI Verificar
	        estado[id]=ESTADOMUERTO; // se mata la mosquita
			}
    	}
};


__global__ void envejecer_kernel(int *estado, int *edad,int *pupacion,int *N_mobil,int dia)
{
    	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

        if(id<N){
            if(dia < 80 || dia > 320){
                if(edad[id] > pupacion[id])edad[id]++;} //ADULTAS
  		    else{
  		        edad[id]++;}
    	}
};

//sumo un dia a cada tacho (cada tacho tiene su reloj para poder descacharrar en forma no sincronica) y resto un dia a dispo tahco

__global__ void delay_kernel(int *devTauTacho,int dia,int *devDiaDelTacho)
{
    
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

        if(id<NUMEROTACHOS){
            if(devTauTacho[id] > 0)devTauTacho[id]=devTauTacho[id] - 1; 
			devDiaDelTacho[id]=dia;
    	}
};


//el siguiente kernel me crea una mascara de los tachos para descacharrar con 1=descacharro
__global__ 
void kernelUnaDim(int *tpd, int *ind, int *disp, bool *mask, int *diadeltacho, int *freq, int *comienzodesc,int dia)
{
    int manzana=threadIdx.x+blockIdx.x*blockDim.x;

    if(manzana<NUMEROMANZANAS){
        int tachoi=ind[manzana];
        int tachof=(manzana+1<NUMEROMANZANAS)?ind[manzana+1]:NUMEROTACHOS;
        int faltandescacharrar=tpd[manzana];
        
        RNG philox;         
	RNG::ctr_type c={{}};
	RNG::key_type k={{}};
	RNG::ctr_type r;
	k[0]=manzana; 
	c[1]=dia;
	c[0]=SEMILLAGLOBAL; 
		
    	r = philox(c, k); 
    	double azar;
    	azar=(u01_closed_open_32_53(r[0]));//numero aleatorios entre [0,1)
    	
        for(int tacho=tachoi;tacho<tachof && faltandescacharrar>0;tacho++){
			//if dia del tacho modulo freq =0 me fijo si esta disponible y lo marco con 1 para descacharrar
			//le agrego innicialmente una no sincronizacion empezando a descacharrar un dia aleatorio entre 1 y 7
			if( (diadeltacho[tacho] + comienzodesc[tacho]) %freq[tacho]==0) //para que no esten sincronizados
			{
            	if(disp[tacho]==0) //el tacho esta disponible
				{
                	mask[tacho]=1;
                	faltandescacharrar--;
					disp[tacho]=nTau;//1 + azar*10;
				}
			}else{mask[tacho]=0;}
        }
    }
}




// functorcito para transferencia de tacho

struct transferirdetacho{
	int m;
	int tpupad;
	int* ptr;
	int cuantos;
	int dia;

	transferirdetacho(int m_,int tpupad_,int *ptr_,int cuantos_, int dia_):
	m(m_),tpupad(tpupad_),ptr(ptr_),cuantos(cuantos_), dia(dia_){};

	__device__ 
	int operator()(thrust::tuple<int,int> tup, int counter){
		int tachoactual=thrust::get<0>(tup);
		int edad=thrust::get<1>(tup);


		int tachonuevo=tachoactual;

		if(tachoactual==m && edad>=tpupad)
		{
			/* sortear nuevo tacho de la misma manzana*/
			RNG philox;         
			RNG::ctr_type c={{}};
			RNG::key_type k={{}};
			RNG::ctr_type r;
			k[0]=counter; 
			c[1]=dia;
			c[0]=SEMILLAGLOBAL; 

			r = philox(c, k); 

			float azar=(u01_closed_closed_32_53(r[0]));//generador de números aleatorios en el device entre [0,1]

			int indicedetachoelegido=int(azar*cuantos);

			
			//tachonuevo=tachoactual; // este es solo para test


			if(indicedetachoelegido<cuantos){
			tachonuevo=ptr[indicedetachoelegido];
			}

		}
		return tachonuevo;
	}
};


// functorcito  para generar randoms uniformes

struct uniformRanInt{
	int dia;
	int medio;
	int ancho;
	uniformRanInt(int medio_, int ancho_, int dia_):medio(medio_),ancho(ancho_),dia(dia_)
	{};

	__device__ int operator()(int i)
	{
	 	RNG philox;         
	    RNG::ctr_type c={{}};
	    RNG::key_type k={{}};
	    RNG::ctr_type r;
	    k[0]=i; 
	    c[1]=dia;
	    c[0]=SEMILLAGLOBAL; 

		r = philox(c, k); 
     	float azar=(u01_closed_open_32_53(r[0]));//numero aleatorios entre [0,1)
		return int(medio+ancho*azar);
	}
};	

// otro functorcito usado para las estadisticas desagregadas
struct acuaticoeneltacho{
	int m;
    int t;
	acuaticoeneltacho(int m_, int t_):m(m_),t(t_){};
    
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int tach=thrust::get<0>(tupla);
        int edad=thrust::get<1>(tupla);
		return (tach==m && edad<t);
	}
};

struct aereaeneltacho
{
	int m;
    int t;
	aereaeneltacho(int m_, int t_):m(m_),t(t_){};   
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int tach=thrust::get<0>(tupla);
        int edad=thrust::get<1>(tupla);
		return (tach==m && edad>t);
	}
};

// NUEVO: functorcito para contar la población total por manzana
struct vivos_x_manzana{
    int m;
	vivos_x_manzana(int m_):m(m_){};   
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int estado=thrust::get<0>(tupla); //NUEVO:cambie el orden de la tupla
        int manzana=thrust::get<1>(tupla);
		return (estado == ESTADOVIVO && manzana == m);
	}
};

//NUEVO: functorcito para contar acuáticos por manzana
struct acuaticos_x_manzana{
    int m;
	acuaticos_x_manzana(int m_):m(m_){};   
	
	__device__ bool operator()(thrust::tuple<int,int,int> tupla)
	{
        int edad=thrust::get<0>(tupla); //NUEVO:cambie el orden de la tupla
        int pupacion=thrust::get<1>(tupla);
        int manzana=thrust::get<2>(tupla);
		return (edad < pupacion && manzana == m);
	}
};
//NUEVO: functorcito para contar adulto por manzana
struct adultos_x_manzana{
    int m;
	adultos_x_manzana(int m_):m(m_){};   
	
	__device__ bool operator()(thrust::tuple<int,int,int> tupla)
	{
        int edad=thrust::get<0>(tupla); 
        int pupacion=thrust::get<1>(tupla);
        int manzana=thrust::get<2>(tupla);
		return (edad >= pupacion && manzana == m);
	}
};


//functorcito para contar adultos en la población
struct poblacion_1{
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int edad=thrust::get<0>(tupla);//NUEVO: cambie el orden de la tupla
        int pupacion=thrust::get<1>(tupla);
		return (edad >= pupacion);
	}
};

// functorcito para contar acuáticos en la población
struct poblacion_2{
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int edad=thrust::get<0>(tupla); //NUEVO:cambie el orden de la tupla
        int pupacion=thrust::get<1>(tupla);
		return (edad < pupacion);
	}
};

struct bichos{

	thrust::host_vector<int> tach;
	
	
	//defino arrays grandes en device la  info de cada mosquita. Numero_mosquitas elementos
	thrust::device_vector<int> estado;  //viva o muerta 0/1
	thrust::device_vector<int> edad;    //edad de la mosquita   
	thrust::device_vector<int> tacho;   // numero de tacho en que se encuentra cada mosquita valores=0 a NUMEROTACHOS  
	thrust::device_vector<int> TdV;     //tiempo de vida de cada mosquita
	thrust::device_vector<int> pupacion; //dia de paso de pupa a adulta de cada mosquita
	thrust::device_vector<int> manzana; //nro. de manzana de cada mosquita
	thrust::device_vector<int> manzdisp;


	// arrays medianos en device, numero_tachos elementos
	thrust::device_vector<int> nacidos; // tiene el num de tachos elementos, numero de nacidos por tacho
	thrust::device_vector<int> devTauTacho; //NUEVOKARI tiene la dispo del tacho 0 disponible, no cero no disponible
	thrust::device_vector<int> devDiaDelTacho;//NUEVOKARI vector en device con numero de tachos elementos: transcurrir de los dias en cada tacho (esto permite descacharrar en dias diferentes)
    	thrust::device_vector<int> devFreqDesc;//NUEVOKARI frecuencia en dias con la que se descacharra cada tacho
	//Ejemplo: si esta fija en 7, todos los tachos se descacharran cada 7 dias. aleatorio alrededor de 7 
	//algunos descacharran cada 5 otros cada 8 etc.
	thrust::device_vector<int> devStartDesc;//NUEVOKARI cada tacho comienza a descacharrar un dia al azar de la semana para evitar sincronia


	// arrays grandes en device, numero de manzanas elementos
	thrust::device_vector<int> devTachosManzana; //NUEVOKARI vector en device con numero manzanas elementos: numero de tachos por manzana
	thrust::device_vector<float> devEpropManzana; //NUEVOKARI vector en device con numero manzanas elementos: efectividad propaganda por manzana
	thrust::device_vector<int> devDescachManzana;// Num de tachos a descacharrar por manzana
	
	thrust::device_vector<int> devIndicesDescach; // NUEVOKARIindice primer tacho de cada manzana
    	thrust::device_vector<bool> mask; //

	// arrays medianos en host, numero_manzanas elementos
	std::vector<std::vector<int> > tachos_por_manzana; //tachos_por_manzana[i]=vector de tachos de manzana i 

	// numero de tachos elementos
	std::vector<int> manzana_del_tacho;
	

	//array para almacenar nro de tachos por manzana
    	thrust::device_vector<int> NroTachos;
	
	// array de un elemento = device variable
	thrust::device_vector<int> N_mobil; // Numero de bichos fluctuante (1 elemento)

	// punteros crudos a los arrays para pasarselos a kernels
	int *raw_edad;
	int *raw_tacho;
	int *raw_estado;
	int *raw_TdV;
	int *raw_pupacion;
	int *raw_N_mobil;
	int *raw_manzana;
	int *raw_nacidos;
	int *raw_devTauTacho;
	int *raw_devDiaDelTacho;
	bool *raw_mask;

	//constructor	
	bichos(int N_,long *semilla){

		// alocamos el maximo posible
		estado.resize(MAXIMONUMEROBICHOS);	
		tacho.resize(MAXIMONUMEROBICHOS);	
		edad.resize(MAXIMONUMEROBICHOS);
		pupacion.resize(MAXIMONUMEROBICHOS);
		TdV.resize(MAXIMONUMEROBICHOS);
		manzana.resize(MAXIMONUMEROBICHOS);
		manzana_del_tacho.resize(MAXIMONUMEROBICHOS);		

		N_mobil.resize(1);
		
		tachos_por_manzana.resize(NUMEROMANZANAS);
		devTachosManzana.resize(NUMEROMANZANAS);
		devEpropManzana.resize(NUMEROMANZANAS); 
		devDescachManzana.resize(NUMEROMANZANAS);
		devIndicesDescach.resize(NUMEROMANZANAS);
		manzdisp.resize(NUMEROMANZANAS);



		NroTachos.resize(NUMEROMANZANAS);

		devTauTacho.resize(NUMEROTACHOS);//NUEVOKARI vector en device de dim numero de tachos cada elemento indica la disponibiilidad del tacho
		mask.resize(NUMEROTACHOS);//NUEVOKARI vector que crea un vector "mascara" con el numero de tacho que hay que descacharrar 1=descacharra
		tach.resize(MAXIMONUMEROBICHOS); //para que funcione bichos.cu

		devDiaDelTacho.resize(NUMEROTACHOS);//NUEVOKARI vector en device de dim numero de tachos dia de cada tacho
		devFreqDesc.resize(NUMEROTACHOS);//NUEVOKARI cada cuantos descacharra cada tacho
		devStartDesc.resize(NUMEROTACHOS);//NUEVOKARI cuando empiezo a descacharrar cada tacho

		// nacidos en cada tacho, inicialmente 0
		nacidos.resize(NUMEROTACHOS);
		thrust::fill(nacidos.begin(),nacidos.end(),0);
		thrust::fill(devTauTacho.begin(),devTauTacho.end(),0);   //NUEVOKARI inicializo en cero
		thrust::fill(devDiaDelTacho.begin(),devDiaDelTacho.end(),1);   //NUEVOKARI inicializo en dia 1 
		thrust::fill(devEpropManzana.begin(),devEpropManzana.end(),PROP); //NUEVOKARI vector efectividad de propaganda que puede ser diferente para cada menzana
		thrust::fill(devTachosManzana.begin(),devTachosManzana.end(),0);
		thrust::fill(devDescachManzana.begin(),devDescachManzana.end(),0);
		thrust::fill(devIndicesDescach.begin(),devIndicesDescach.end(),0);
		thrust::fill(mask.begin(),mask.end(),0);
		thrust::fill(manzdisp.begin(),manzdisp.end(),0); //vector que indica si la manzana todavia tiene lugar para poner tachos
														//0 tiene lugar 1=no tiene lugar	

		thrust::fill(estado.begin(),estado.end(),0);
		thrust::fill(edad.begin(),edad.end(),0);
		thrust::fill(tacho.begin(),tacho.end(),0);
		thrust::fill(pupacion.begin(),pupacion.end(),0);
		thrust::fill(TdV.begin(),TdV.end(),0);
		thrust::fill(manzana.begin(),manzana.end(),0);

		// inicializacion raw pointers para pasarlos al kernel		
		raw_edad=thrust::raw_pointer_cast(edad.data());
		raw_tacho=thrust::raw_pointer_cast(tacho.data());
		raw_estado=thrust::raw_pointer_cast(estado.data());
		raw_TdV=thrust::raw_pointer_cast(TdV.data());
		raw_manzana=thrust::raw_pointer_cast(manzana.data());
		raw_devTauTacho=thrust::raw_pointer_cast(devTauTacho.data());
		raw_mask=thrust::raw_pointer_cast(mask.data());
		raw_devDiaDelTacho=thrust::raw_pointer_cast(devDiaDelTacho.data());
	
		raw_N_mobil=thrust::raw_pointer_cast(N_mobil.data());
		raw_pupacion=thrust::raw_pointer_cast(pupacion.data());
		raw_nacidos=thrust::raw_pointer_cast(nacidos.data());

	
//*************************************************************************************************************
//                       condiciones iniciales para N_=NINICIAL ingresado en el archivo parametros.h
//*************************************************************************************************************
		for(int i=0;i < N_;i++){
		    
	    		tacho[i] = i;			                   //tacho en el que se encuentra la mosquita
			if(DISTRIBUCIONTACHOS){
				manzana_del_tacho[tacho[i]]=int (i/5);    //para 5 tachos por manzana
			}
    			else{
	    			manzana_del_tacho[tacho[i]]=int(ran2(semilla)*NUMEROMANZANAS);//le asigno al tacho una manzana al azar 
			//Yo lo del tope de 9 tachos no lo pondria a menos que lo veamos necesario por alguna razon.
				int tachosxmanzana=tachos_por_manzana[manzana[i]].size();      //nro de tachos x manzana
				if(tachosxmanzana > 9){ 
				manzdisp[manzana[i]]=1; //esta manzana ya no esta disponible entonces sorteo entre las demas	
				//thrust::device_vector<int>::iterator iter;
				auto iter=thrust::find(manzdisp.begin(),manzdisp.end(),0); //me tira la posicion de la primer manzana con lugar
				manzana_del_tacho[tacho[i]]=int(iter-manzdisp.begin()); // 
				} 
			}

			manzana[i]=manzana_del_tacho[tacho[i]];            //manzana en la que está el tacho i
			tachos_por_manzana[manzana[i]].push_back(tacho[i]);//para identificar los tachos tengo en la manzana
 	   		devTachosManzana[manzana[i]]++; //incremento en 1 tacho en la manzana correspondiente
			devTauTacho[i]=0; //NUEVOKARI el tacho i esta disponible = 0, estará no disponible cuando no sea cero.
			estado[i] = ESTADOVIVO; 	      		     //todas vivas inicialmente
			edad[i] = ran2(semilla)*7+19; 	             //todas adultas al principio 
			pupacion[i] = TPUPAD-2+(ran2(semilla)*5);          //dia de pupacion (entre los 15 y 19 dias)
			TdV[i] = ran2(semilla)*6+27 ;	                    //tiempo de vida de 27 a 32
					
		N_mobil[0]=N_;
		}

//std::cout << "************************************************************************************" << "\n";
//std::cout << "verificacion del llenado de tachos x manz. y de la disponibilidad de tacho x manz."<< "\n";
//std::cout << "*********************************************|***************************************" << "\n";

/*		int nmanzanas=tachos_por_manzana.size();

		for(int i=0;i<nmanzanas;i++){
			//std::cout << "\n\n manzana " << i << "\n";
			int ntachos=tachos_por_manzana[i].size();   //nro de tachos por manzana
			int contTach=0;                             //contador para el nro de tachos por manzana

			for(int j=0;j<ntachos;j++){
				//std::cout << (tachos_por_manzana[i])[j] << "\n";
				contTach++;
				NroTachos[i]=contTach;  
			}//cierro for para ntachos
		std::cout << i << "\t" << NroTachos[i] << std::endl;
		}//cierro for para nmanzanas
*/		
	};	

//KARINUEVO: esta funcion sirve para que el programa corte cuando se acaban las mosquitas
//si no hay mas mosquitas adultas para reproducirse no tiene sentido que el programa siga calculando
	int getNmobil(){
		if(N_mobil[0]>0) return N_mobil[0]; 
		else{
			std::cout << "no hay mas mosquitas vivas"<< std::endl; std::cout.flush();
			exit(1);
		}
	}


	// devuelve un numero de tacho random de la manzana m
	int nuevo_tacho_misma_manzana(int m){
		int ntachos=tachos_por_manzana[m].size();
		int r=int((rand()*1.0/RAND_MAX)*ntachos);		
		int nuevo_tacho=(tachos_por_manzana[m])[r];
		return nuevo_tacho;
	}
	
	
	// devuelve numero de manzana sorteada entre las cuatro manzanas vecinas de la manzana m
		int sorteo_manzana_vecina(int manz){
		// numero de las manzanas vecinas de una manzana
		std::vector<int> manzanas_vecinas(5);
				
		int x=manz%LADO;
		int y=int(manz/LADO);
		manzanas_vecinas[0]=(x-1+LADO)%LADO+LADO*y; //izqu
		manzanas_vecinas[1]=(x+1+LADO)%LADO+LADO*y; //derecha
		manzanas_vecinas[2]=LADO*((y-1+LADO)%LADO)+x; //abajo
		manzanas_vecinas[3]=LADO*((y+1+LADO)%LADO)+x; //arriba
		manzanas_vecinas[4]=manz; //centro
		
		int nvecinos=6; //cuento los vecinos y el centro
		int r=int((rand()*1.0/RAND_MAX)*nvecinos); //numero aleatorio entre 0 y 5
		//si sale 2,3,4,5 elijo el centro (es una forma trucha de darle mas probabilidad al centro)
		int manzana_sorteada;
		if(r>1)
			{
			manzana_sorteada=manz;
			}
			else{		
			manzana_sorteada=manzanas_vecinas[r];
		}
		return manzana_sorteada;
	}


	void mortalidades(int dia){

	int N=getNmobil();
	//mortalidades varias y muerte por vejez
	    matar_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_pupacion,raw_TdV,raw_N_mobil, dia);	
		
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	};

	void descacharrado(int dia,long *semilla){
	int N=getNmobil();
	    	
		//Solo el primer dia calculo cuantos tachos por manzana voy a descacharrar luego esto queda fijo
		if(dia==1)
		{				
			/*Opté por la siguiente opcion que va con todos los compiladores siempre que se agregue lo siguiente en el makefile
			 compilar con nvcc --expt-extended-lambda*/
			thrust::transform(
			devTachosManzana.begin(),devTachosManzana.end(),devEpropManzana.begin(),
			devDescachManzana.begin(),[=]__device__ (float x,float y){return x*y;}
			);
				
			//tendre el numero de tachos a descacharrar por manzana en devDescachManzana
			int descachTot=0;
			descachTot=thrust::reduce(devDescachManzana.begin(),devDescachManzana.end(),descachTot);
			//el numero de tachos total a descacharrar sera la suma sobre ese vector, lo que llamaba descach
    			
			int ntachos=thrust::reduce(devTachosManzana.begin(),devTachosManzana.end());
			std::cout << "nro total de tachos = " << ntachos << std::endl; 
			//calculo los indices de los tachos a descacharrar (exclusive_scan va acumulando el num de tachos por manzana)
    		thrust::exclusive_scan(devTachosManzana.begin(),devTachosManzana.end(),devIndicesDescach.begin());
    		std::cout << "ntachos,begin,descacharrar" << std::endl; 
    			/* for(int i=0;i<NUMEROMANZANAS;i++){
        			std::cout << devTachosManzana[i] << "," <<  devIndicesDescach[i] << "," << devDescachManzana[i] << std::endl;
    			 }*/
						 
			//El descacharrado puede ser fijo o aleatorio
			//int DiasDesc;
			if(DESCACHFIJO)
			{
			thrust::fill(devFreqDesc.begin(),devFreqDesc.end(),TIEMPODESCACH);
				for(int i=0;i<NUMEROTACHOS;i++){
					devStartDesc[i]=ran2(semilla)*OFFSETDESCACH; //OFFSET para descacharrado no sincrónico
					std::cout << devFreqDesc[i] << ",  Freq descacharrado" << devStartDesc[i] << ",  Comienzo descach" << std::endl;
    			}
			}else
			{
			//Ahora Dias Desc tiene que ser un vector de Ntachos elementos
			//thrust::fill(devFreqDesc.begin(),devFreqDesc.end(),1 + ran2(semilla)*2*TIEMPODESCACH);
			//lo de arriba no estaba bien porque me cambiaba la frecuencia de descacharrado cada dia (antes estaba
			//fuera del loop dia=1)
				for(int i=0;i<NUMEROTACHOS;i++){
					devFreqDesc[i]=1 + ran2(semilla)*2*TIEMPODESCACH;
        			//devStartDesc[i]=121 + ran2(semilla)*6;
					devStartDesc[i]=ran2(semilla)*OFFSETDESCACH; //OFFSET para descacharrado no sincrónico entre 1 y 4 dias
					std::cout << devFreqDesc[i] << ",  Freq descacharrado" << devStartDesc[i] << ",  Comienzo descach" << std::endl;
    			}
			}	
		}


	    if(dia > 120 && dia < 320){
		// el siguiente kernel me saca una mascara con los indices de tachos a descacharrar (todavia no descacharra)
		kernelUnaDim<<<(NUMEROMANZANAS+BLOCKS-1)/BLOCKS,BLOCKS>>>(
        thrust::raw_pointer_cast(devDescachManzana.data()),
        thrust::raw_pointer_cast(devIndicesDescach.data()),
		raw_devTauTacho,
        raw_mask,
		raw_devDiaDelTacho,
		thrust::raw_pointer_cast(devFreqDesc.data()),
		thrust::raw_pointer_cast(devStartDesc.data()),dia
    	);

		//El siguiente kernel me descacharra los tachos que diga la mask
		descacharrado_kernel<<<(N+256-1)/256,256>>>(raw_estado, raw_edad, raw_tacho, raw_pupacion, raw_N_mobil,dia,raw_mask);
  		cudaDeviceSynchronize();
		
		}

   	};

    // //nacimientos
	void reproducir(int dia,int tovip)
	{
	    
		int indice=getNmobil();
		if(indice==0) {
			std::cout << "NO HAY MAS MOSQUITAS PARA REPRODUCIRSE" << std::endl; 	

		exit(1);//comenté esta linea porque terminaba el programa y no era necesario
		}else{

		//nacimientos
		//antes de reproducir reinicializo en cero los nacidos en el paso anterior que ahora ya no son mas nacidos porque crecieron 
		thrust::fill(nacidos.begin(),nacidos.end(),0);

		// reproduce primero y luego ve si los tachos no están saturados para poner los huevos
		kernel_reproducir<<<(indice+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_TdV,raw_pupacion,raw_manzana,raw_N_mobil,dia,tovip,raw_nacidos);
		cudaDeviceSynchronize();

			// despues de reproducir agrego todos los nacidos al final del array original, tacho a tacho
			int index=indice;
			
			for(int m=0;m<NUMEROTACHOS;m++)
			{

				//calculo el nunmero de acuaticos en cada tacho
				int antiguos=thrust::count_if(
					thrust::make_zip_iterator(thrust::make_tuple(tacho.begin(),edad.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(tacho.begin()+indice,edad.begin()+indice)),
					acuaticoeneltacho(m,TPUPAD)
				);


				//los nuevos vienen del kernel reproducir  
				int nuevos=nacidos[m];
			        
				    //Ahora bien, si con los nuevos supero el maximo de huevos por tacho (SAT)y (NUEVO) el tacho está disponible
				if(nuevos+antiguos>SAT && devTauTacho[m]==0){    
					nuevos=SAT-antiguos;	//ponen lo que pueden en el mismo tacho
					//nuevos=0; //para imitar lo que hace Fabi que no llena el tacho cuando ve que se va a pasar de la saturacion
					if(TRANSFERTACHO==1) //transfiero de tacho en misma manzana
					{
						int manzanadeltacho;


						if(TRANSFERMANZANA==1)
						{
						/*Para transferir de tacho y de manzana*/
						/*Muevo LOS ADULTOS a otro tacho de la misma manzana o de una manzana vecina*/
						int estamanzana = manzana_del_tacho[m];
						manzanadeltacho = sorteo_manzana_vecina(estamanzana); //pone en la misma manzana o en una vecina
						}else //ovipone en la misma manzana
						{
						manzanadeltacho = manzana_del_tacho[m]; //se queda en esta manzana
						}

					int cuantos=(tachos_por_manzana[manzanadeltacho]).size(); //nro de tachos por manzana

					int* ptr_h=(tachos_por_manzana[manzanadeltacho]).data();

					gpu_timer relojito;
					relojito.tic();
					thrust::device_vector<int> tachosDeLaManzana(cuantos);

						for(int k=0;k<cuantos;k++){
							tachosDeLaManzana[k]=ptr_h[k];
						}	
						//std::cout << "ineficiente?=" << relojito.tac() << std::endl;
							
					int* ptr_d=thrust::raw_pointer_cast(tachosDeLaManzana.data());

					//el siguiente transform me tira el numero de tacho donde va a oviponer 
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(tacho.begin(),edad.begin())),
						thrust::make_zip_iterator(thrust::make_tuple(tacho.begin()+indice,edad.begin()+indice)),
						thrust::make_counting_iterator(0),
						tacho.begin(),
						transferirdetacho(m,TPUPAD,ptr_d,cuantos, dia)
					);
					
						

					}	
				//Una vez que se llenaron los tachos de la manzana, pone en las manzanas vecinas

				}//cierro if para transferencia de tacho
				

				/*HASTA ACÁ MAYOR PROBABILIDAD DE TRANSFERENCIA DE TACHO EN LA MISMA MANZANA Y MENOR PORB. DE TRANSFERENCIA DE MANZANA y TACHO */
				
				if(devTauTacho[m]==0)
				{ //KARINUEVO m ahora es el tacho nuevo
					thrust::fill(estado.begin()+index,estado.begin()+index+nuevos,ESTADOVIVO);	//nacen todas vivas       
					thrust::fill(edad.begin()+index,edad.begin()+index+nuevos,1);		        //nacen con edad(dias)      
					thrust::fill(tacho.begin()+index,tacho.begin()+index+nuevos,m); 	        //nacen en el tacho m

					
					// index en counting iteraror necesario para distintos randoms en cada tacho
					thrust::transform(
						thrust::make_counting_iterator(index),thrust::make_counting_iterator(index+nuevos),
						pupacion.begin()+index,uniformRanInt(15,5,dia)
					);
				
					
					thrust::transform(
						thrust::make_counting_iterator(index),thrust::make_counting_iterator(index+nuevos),
						TdV.begin()+index,uniformRanInt(27,6,dia)
					);

					thrust::fill(manzana.begin()+index,manzana.begin()+index+nuevos,manzana_del_tacho[m]);        
					index+=nuevos;		//actualizo el indice para me marque siempre en la ultima mosquita que nacio    

                		}

			}//cierro for para los tachos
		
		    // actualiza el indice movil hasta el ultimo bicho vivo

			if(index<MAXIMONUMEROBICHOS) {
				N_mobil[0]=index;
			}
			else{ ////satura la memoria reservada salgo del prog
				std::cout << "Demasiados Bichos!" << std::endl;
				exit(1);
			}	
				
		}//cierro else linea 523	
		
	};

//Recalcular -> eliminar muertos y dejar vivos
    void recalcularN(){
	    
		auto zip_iterator=
		thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),tacho.begin(),pupacion.begin(),TdV.begin(),manzana.begin()));
		// ordenamos segun estado 0-vivo, 1-muerto
		int N=getNmobil();
		thrust::sort_by_key(estado.begin(), estado.begin() + N,zip_iterator);		
	
		// y ahora determinamos la posicion del primer muerto = N_mobil
		auto iter=thrust::find(estado.begin(),estado.begin() + N, ESTADOMUERTO);
		N_mobil[0]= iter-estado.begin();//me da la longitud del vector
		//std::cout << "N_mobil " << N_mobil[0] <<std::endl;
	};
	
	//población total adultos + acuáticos
	int vivos(int dia){

	int N=getNmobil();
   	int poblacion = thrust::count(estado.begin(), estado.begin() + N, ESTADOVIVO);
	return poblacion;
	};

    //población de acuáticos
	int acuaticos(int dia){
		
	int N=getNmobil();	
    	int ac= thrust::count_if(
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),pupacion.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin() +  N,pupacion.begin() + N)),
                poblacion_2()
            );
	return ac;
	};

    //población de adultos
	int adultos(int dia){

	int N=getNmobil();
	//el predicado poblacion_1()corresponde a adultos
	    int ad=thrust::count_if(
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),pupacion.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin() +  N,pupacion.begin() + N)),
                poblacion_1()
            );

		return ad;
	};	

	//envejecer población
   	void envejecer(int dia){
	int N=getNmobil();
        envejecer_kernel<<<(N + 256-1)/256,256>>>(raw_estado,raw_edad,raw_pupacion,raw_N_mobil,dia);
        cudaDeviceSynchronize();
	}; 
	
	//NUEVOKARI disminuir el delay=nTau en 1 día para que el tacho eliminado vuelva a estar disponible 
	//y llevo cuenta del dia del tacho para poder descacharrar no sincronico
   	void delay(int dia){
	int N=getNmobil();
	delay_kernel<<<(N + 256-1)/256,256>>>(raw_devTauTacho,dia,raw_devDiaDelTacho);
        cudaDeviceSynchronize();
       	
	}; 
	
	void output(int dia){
	int N=getNmobil();

	int nmanzanas=tachos_por_manzana.size();
		for(int i=0;i<nmanzanas;i++){
		        int conMosq= thrust::count_if(
	                thrust::make_zip_iterator(thrust::make_tuple(estado.begin(),manzana.begin())),
	                thrust::make_zip_iterator(thrust::make_tuple(estado.begin() +  N,manzana.begin() + N)),
	                vivos_x_manzana(i)
		            );

		        int conAd= thrust::count_if(
	                thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),pupacion.begin(),manzana.begin())),
	                thrust::make_zip_iterator(thrust::make_tuple(edad.begin() +  N,pupacion.begin()+N,manzana.begin() + N)),
	                adultos_x_manzana(i)
		            );

		        int conAc= thrust::count_if(
        	        thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),pupacion.begin(),manzana.begin())),
        	        thrust::make_zip_iterator(thrust::make_tuple(edad.begin() +  N,pupacion.begin()+N,manzana.begin() + N)),
        	        acuaticos_x_manzana(i)
	            );

        	fprintf(archivo1,"%d\t%d\t%d\n",dia,i,conMosq);
        	fprintf(archivo2,"%d\t%d\t%d\n",dia,i,conAd);
        	fprintf(archivo3,"%d\t%d\t%d\n",dia,i,conAc);
		}//cierro for para nmanzanas

	fprintf(archivo1,"\n");
	fprintf(archivo2,"\n");
	fprintf(archivo2,"\n");
	};
	
};

int main(int argc,char **argv){

    thrust::host_vector <int> PoblacionT;           //Población de mosquitas Total= Adultas+acuáticas
    thrust::host_vector <int> PoblacionAd;          //Población de mosquitas Adultas
    thrust::host_vector <int> PoblacionAc;          //Población de mosquitas Acuáticas
    
    PoblacionT.resize(NDIAS+1);	
    PoblacionAd.resize(NDIAS+1);
    PoblacionAc.resize(NDIAS+1);
	
    //inicializo los arrays en 0
    thrust::fill(PoblacionT.begin(),PoblacionT.end(),0);
    thrust::fill(PoblacionAd.begin(),PoblacionAd.end(),0);
    thrust::fill(PoblacionAc.begin(),PoblacionAc.end(),0);
    
    //loop para el número de corridas con distinta semilla
	
    for(int seed=0;seed<NITERACIONES;seed++){
    std::cout << "nro de realizacion: "<< seed+1 << "\n";
    //incializamos semilla
    long semilla = -739 + 100;
    //long semilla= -739 + atoi(argv[1]); //cuando se mandan múltiples corridas al cluster + using jobGPUtest in Makefile
    
    gpu_timer Reloj_GPU;
    Reloj_GPU.tic();
    
    //inicializo
    bichos mosquitas(NINICIAL,&semilla);
	    
    //Guardo población total por manzana
    sprintf(miarch,"manzanas_vs_Nro-de-mosquitas_%d.txt",seed);
    archivo1=fopen(miarch,"w");
    //guardo poblacion adultas por manzana
    sprintf(miarch,"manzanas_vs_Nro-de-adultas_%d.txt",seed);
    archivo2=fopen(miarch,"w");
    //guardo poblacion acuaticas por manzana
    sprintf(miarch,"manzanas_vs_Nro-de-acuaticas_%d.txt",seed);
    archivo3=fopen(miarch,"w");
    //Guardo población total por dia
    sprintf(miarch,"POBLACION-T_%d.txt",seed);
    archivoT=fopen(miarch,"w");
    //Guardo población adulta por dia
    sprintf(miarch,"POBLACION-Ad_%d.txt",seed);
    archivoAd=fopen(miarch,"w");
    //Guardo población acuatica por dia
    sprintf(miarch,"POBLACION-Ac_%d.txt",seed);
    archivoAc=fopen(miarch,"w");


    double treprodparalelo=0;    
    double trecalc=0;
    double tdescacha=0;
    
        //calculo la población en cada iteración
	    for(int dia = 1; dia <= NDIAS; dia++){
	    std::cout << "DIA" << dia << std::endl;
            int tovip=tiempo_entre_oviposiciones(dia);
	
	    //std::cout << "matar" << std::endl;
	    mosquitas.mortalidades(dia);
	    
	    gpu_timer Reloj_descacharrar;
	    Reloj_descacharrar.tic();
	    //std::cout << "descacharrar" << std::endl;
	    mosquitas.descacharrado(dia,&semilla); 
	    tdescacha= tdescacha+Reloj_descacharrar.tac()/60000; //de milisegundos -> minutos

	    gpu_timer Reloj_reproducirparalelo;
	    Reloj_reproducirparalelo.tic();
	    //std::cout << "reproducir" << std::endl;
	    mosquitas.reproducir(dia,tovip); //version paralela tarda mucho
	    treprodparalelo= treprodparalelo+Reloj_reproducirparalelo.tac()/60000; //de milisegundos -> minutos
		
	    gpu_timer Reloj_recalcular;
	    Reloj_recalcular.tic();
	    //std::cout << "recalcular indice de mosquitas vivas" << std::endl;
	    mosquitas.recalcularN(); 
	    trecalc= trecalc+Reloj_recalcular.tac()/60000; //de milisegundos -> minutos
	
            //en esta versión del programa solo considero hembras
	    int vivas=mosquitas.vivos(dia);
	    int adultas=mosquitas.adultos(dia);
	    int acuaticas=mosquitas.acuaticos(dia);

	    //std::cout << "envejecer poblacion" << std::endl;
	    mosquitas.envejecer(dia);
	    mosquitas.delay(dia); //(NUEVO) disminuyo en un dia el delay=nTau de los tachos que fueron vaciados
	    PoblacionT[dia]= vivas;//guardo en un vector el nro de mosquitas para una determinada semilla
            PoblacionAd[dia]= adultas;
            PoblacionAc[dia]= acuaticas;
            mosquitas.output(dia);//salida de multiples archivos manzanas vs T, manzanas vs Ad, manzanas vs Ac
            fprintf(archivoT,"%d\t%d \n",dia,PoblacionT[dia]);
	    fprintf(archivoAd,"%d\t%d \n",dia,PoblacionAd[dia]);
	    fprintf(archivoAc,"%d\t%d \n",dia,PoblacionAc[dia]);
	    }//cierro loop para dias

       //tiempos de cáculo en GPU***************************************************		
       double t=Reloj_GPU.tac()/60000; //de milisegundos -> minutos
       printf("Tiempo de cálculo total para Población total de mosquitas en GPU: %lf minutos\n",t);
       printf("Tiempo en descacharrar: %lf minutos\n",tdescacha);
       printf("Tiempo en reproducir paralelo: %lf minutos\n",treprodparalelo);
//     printf("Tiempo en reproducir serial: %lf minutos\n",treprodserial);
       printf("Tiempo en recalcular: %lf minutos\n",trecalc);	
       //****************************************************************************
       std::cout << "\n";
        //miarch es un string de caracteres donde guardo el nombre del archivo cambiando la semilla con cada iteracion
        
        //Guardo población total
	fclose(archivoT); 
	fclose(archivoAd); 
	fclose(archivoAc); 
    }//cierro loop para el número de ITERACIONES
    
return 0;		


}// end for main
