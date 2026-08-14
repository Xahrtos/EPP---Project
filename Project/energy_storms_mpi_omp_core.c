#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "energy_storms.h"

/* 
 * FUNZIONE AUX: update
 * Calcola l'energia che un punto dell'array (k) assorbe quando una particella
 * cade in una certa posizione (pos). L'energia cala con la distanza.
 */
static inline void update( float *layer, int layer_size, int k, int pos, float energy ) {
    int distance = pos - k;
    if ( distance < 0 ) distance = - distance; // Calcola la distanza in valore assoluto
    distance = distance + 1;

    float atenuacion = sqrtf( (float)distance ); // Attenuazione basata sulla radice della distanza
    float energy_k = energy / layer_size / atenuacion;

    // Aggiunge energia al punto 'k' solo se supera una certa soglia minima (THRESHOLD)
    if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
        layer[k] = layer[k] + energy_k;
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    
    // ----------------------------------------------------
    // INIZIALIZZAZIONE MPI
    // ----------------------------------------------------
    int rank, size;
    // 'rank' è l'ID di questo specifico processo (es. Processo 0, 1, 2...)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // 'size' è il numero totale di processi MPI che stanno lavorando insieme
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. ALLOCAZIONE LOCALE DELLA MEMORIA
    // Ogni processo crea la sua copia locale dell'array 'layer' e di 'layer_copy'
    // 'calloc' azzera automaticamente tutta la memoria appena allocata.
    float *layer = (float*) calloc(layer_size, sizeof(float));
    float *layer_copy = (float*) calloc(layer_size, sizeof(float));

    // 2. DECOMPOSIZIONE DEL DOMINIO (Dividere la superficie tra i processi)
    // Immagina che la superficie sia una striscia divisa in tante celle.
    // Dobbiamo distribuire equamente le celle tra i vari processi MPI.
    int chunk_size = layer_size / size; // Quante celle spettano a ciascun processo
    int remainder  = layer_size % size; // Celle avanzate dalla divisione

    // Calcoliamo da quale indice (my_start_k) a quale indice (my_end_k) deve lavorare QUESTO processo
    int my_start_k = rank * chunk_size + (rank < remainder ? rank : remainder);
    int my_count   = chunk_size + (rank < remainder ? 1 : 0);
    int my_end_k   = my_start_k + my_count;

    // Identifichiamo i "vicini di casa" di questo processo lungo la striscia
    // Se sono il rank 0, non ho vicini a sinistra (MPI_PROC_NULL).
    int left_neighbor  = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    int i, j, k;

    // Struttura usata per inviare sia il VALORE massimo che la sua POSIZIONE
    struct {
        float val;
        int pos;
    } in_struct, out_struct;

    // ----------------------------------------------------
    // SIMULAZIONE DELLE TEMPESTE (Ciclo Principale)
    // ----------------------------------------------------
    for (i = 0; i < num_storms; i++) {

        // ====================================================
        // FASE 4.1: BOMBARDAMENTO DI PARTICELLE
        // ====================================================
        // Scorriamo tutte le particelle che cadono durante questa tempesta
        for (j = 0; j < storms[i].size; j++) {
            
            float energy = (float)storms[i].posval[j*2+1] * 1000.0f; // Conversione dell'energia
            int position = storms[i].posval[j * 2];                // Punto di impatto

            // PARALLELISMO OPENMP:
            // Dividiamo il ciclo 'for' tra i vari THREAD (core CPU) dello stesso processo MPI.
            // 'schedule(static)' assegna blocchi uguali di celle ai vari thread.
            #pragma omp parallel for schedule(static)
            for (k = my_start_k; k < my_end_k; k++) {
                update(layer, layer_size, k, position, energy);
            }
        }

        // ====================================================
        // FASE 4.2: RILASSAMENTO (STENCIL)
        // ====================================================
        // Passo A: Copiamo l'array originale in 'layer_copy'
        // Lo facciamo usando più thread in parallelo con OpenMP
        #pragma omp parallel for schedule(static)
        for (k = my_start_k; k < my_end_k; k++) {
            layer_copy[k] = layer[k];
        }

        // Variabili per ospitare i dati provenienti dai processi vicini
        float ghost_left_in = 0.0f;
        float ghost_right_in = 0.0f;

        // HALO EXCHANGE (SCAMBIO DEI BORDI CON MPI)
        // Per aggiornare la mia prima cella (my_start_k), ho bisogno di sapere quanto vale
        // la cella del mio vicino di SINISTRA. E viceversa per la mia ultima cella a DESTRA.
        
        // Invia il mio bordo sinistro a 'left_neighbor' e ricevi il bordo da 'right_neighbor'
        MPI_Sendrecv(&layer_copy[my_start_k], 1, MPI_FLOAT, left_neighbor, 0,
                     &ghost_right_in, 1, MPI_FLOAT, right_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Invia il mio bordo destro a 'right_neighbor' e ricevi il bordo da 'left_neighbor'
        MPI_Sendrecv(&layer_copy[my_end_k - 1], 1, MPI_FLOAT, right_neighbor, 1,
                     &ghost_left_in, 1, MPI_FLOAT, left_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // CALCOLO DELLO STENCIL (Media tra la cella, il vicino sinistro e il vicino destro)
        #pragma omp parallel for schedule(static)
        for (k = my_start_k; k < my_end_k; k++) {
            
            // Se la cella si trova al confine del mio processo, usiamo il dato arrivato via MPI (ghost)
            float prev_val = (k == 0) ? layer_copy[0] 
                            : ((k == my_start_k) ? ghost_left_in : layer_copy[k - 1]);
            
            float next_val = (k == layer_size - 1) ? layer_copy[layer_size - 1] 
                            : ((k == my_end_k - 1) ? ghost_right_in : layer_copy[k + 1]);

            // Calcola la media dei 3 punti
            layer[k] = (prev_val + layer_copy[k] + next_val) / 3.0f;
        }

        // ====================================================
        // FASE 4.3: RICERCA DEL MASSIMO LOCALE E GLOBALE
        // ====================================================
        float local_max_val = -1.0f;
        int local_max_pos = -1;

        // Apriamo una regione parallela OpenMP
        #pragma omp parallel
        {
            // Ogni thread mantiene il SUO massimo locale temporaneo
            float thread_max_val = -1.0f;
            int thread_max_pos = -1;

            // Dividiamo le celle da analizzare tra i thread
            #pragma omp for schedule(static)
            for (k = my_start_k; k < my_end_k; k++) {
                if (layer[k] > thread_max_val) {
                    thread_max_val = layer[k];
                    thread_max_pos = k;
                }
            }

            // CRITICAL: Un solo thread alla volta aggiorna il massimo del processo MPI
            // per evitare che si pestimino i piedi a vicenda (Race Condition).
            #pragma omp critical
            {
                if (thread_max_val > local_max_val) {
                    local_max_val = thread_max_val;
                    local_max_pos = thread_max_pos;
                }
            }
        }

        // Prepariamo i dati per MPI
        in_struct.val = local_max_val;
        in_struct.pos = local_max_pos;

        // REDUCE GLOBALE (MPI_Allreduce)
        // Tutti i processi MPI confrontano il loro massimo locale.
        // 'MPI_MAXLOC' trova il valore massimo ASSOLUTO su tutto l'array globale 
        // e lo distribuisce a tutti i processi dentro 'out_struct'.
        MPI_Allreduce(&in_struct, &out_struct, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        // Salviamo il massimo di questa tempesta 'i'
        maximum[i] = out_struct.val;
        positions[i] = out_struct.pos;
    }

    // Liberiamo la memoria allocata all'inizio
    free(layer);
    free(layer_copy);
}