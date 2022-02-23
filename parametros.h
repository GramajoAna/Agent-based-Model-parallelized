
/*
Macros utiles para CPU y GPU
Relaciones a tener en cuenta para modificar parรกmetros
- NUMEROTACHOS=NINICIAL
*/
//Parámetros generales
#define NINICIAL 		125 //12500	        //número inicial de mosquitos N=125 para 5 mosquitas x manzana.
#define NUMEROTACHOS		125// 12500          //máximo número de tachos Ntachos=Nmosquitas
#define MAXIMONUMEROBICHOS	90000000    //número maximo de huevos
#define LADO           	5 //1000          //Lado de la grilla L=5
#define NUMEROMANZANAS		(LADO*LADO)   //número de manzanas que sea L*L=5*5=25

//#define NUMERODEHUEVOS		10 		//número de huevos por oviposicion	

//Parámetros del descacharrado
#define BLOCKS             32 //kernel de descacharrado
#define DESCACHFIJO        1 //descacharrado fijo=1, aleatorio=0
#define TIEMPODESCACH      7 //cada 7 dias (fijo o aleatorio con distribucion uniforme y media igual a este valor)
#define nTau               1 //tiempo que tardan en estar disponibles los tachos nuevamente
#define DISTRIBUCIONTACHOS 0 // =0 para distrib aleatoria de hasta 9 tachos por manzana y =1 para una distribución de 5 tachos por manzana 
#define TRANSFERTACHO      1 //=1 con transferencia de tacho y 0 sin transferencia de tacho
#define TRANSFERMANZANA    1 // 1= con transferencia de manzana a tachos de manzanas vecinas, 0 se queda en la misma manzana    
#define OFFSETDESCACH      3 //numero al azar de dias de desfasaje en el descacharrado de los tachos entre 0 y este tiempo 

//Parámetros vitales de las mosquitas
#define ESTADOMUERTO		1
#define ESTADOVIVO	    	0
#define NDIAS		    	400	        //nro de dias en el año
//#define NREALIZACIONES	5	        //nro de realizaciones para promediar poblaciones con diferente semillas
#define NITERACIONES   	1	        //nro de realizaciones para promediar poblaciones con diferente semillas

#define MORHUE 		0.01   	    //mortalidad de huevos
#define MORLAR 		0.01       //mortalidad de larvas 
#define MORPUP 		0.01   	    //mortalidad de pupas
#define MORAD 	    		0.01 	    //mortalidad diaria adultas
#define MORACU 	        0.03	    //morhue+morlar+morpup;
#define MORPUPAD 	    	0.17	    //pupas que no se vuelven adultas
#define TPUPAD	    		17  	    //pupas se vuelven adultas a los 17 dias en invierno****
#define TOVIP1 	        2	        //tiempo entre dos oviposiciones (T=30)
#define TOVIP2a     		3  	      //tiempo entre dos oviposiciones (T=25)
#define TOVIP2b		4   	          //tiempo entre dos oviposiciones (T=25)
#define TOVIP3 		30	            //tiempo ente dos oviposiciones (T=18)
#define SAT 		       800        //saturación de huevos por tacho
#define PROP 		    	0.4         //efectividad de la propaganda

#define TPUPAD  	    	17  	    //pupas se vuelven adultas a los 17 dias en invierno****

#define SEMILLAGLOBAL	   	123456789 //para el generador de números aleatorios en el device PHILOX
