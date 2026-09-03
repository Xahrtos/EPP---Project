#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "energy_storms.h"

/* 
 * FUNZIONE AUX: update
 * Calcola l'energia assorbita dal punto 'k' dell'array al cadere di una particella in 'pos'.
 */
static inline void update(float *layer, int layer_size, int k, int pos, float energy) {
    int distance = pos - k;
    if (distance < 0) distance = -distance;
    distance = distance + 1;

    float atenuacion = sqrtf((float)distance); 
    float energy_k = energy / layer_size / atenuacion;

    if (energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size)
        layer[k] = layer[k] + energy_k;
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    
    // ----------------------------------------------------
    // INIZIALIZZAZIONE MPI
    // ----------------------------------------------------
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. ALLOCAZIONE MEMORIA LOCALE (Inizializzata a zero)
    float *layer = (float*) calloc(layer_size, sizeof(float));
    float *layer_copy = (float*) calloc(layer_size, sizeof(float));

    if (layer == NULL || layer_copy == NULL) {
        fprintf(stderr, "Error: Allocating memory for layer\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // 2. DECOMPOSIZIONE DEL DOMINIO
    int chunk_size = layer_size / size; 
    int remainder  = layer_size % size; 

    // Range di calcolo locale per il processo MPI corrente
    int my_start_k = rank * chunk_size + (rank < remainder ? rank : remainder);
    int my_count   = chunk_size + (rank < remainder ? 1 : 0);
    int my_end_k   = my_start_k + my_count;

    // Identificazione dei vicini MPI per lo scambio Ghost Cells
    int left_neighbor  = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // Struttura MPI per la riduzione del massimo con posizione
    struct {
        float val;
        int pos;
    } in_struct, out_struct;

    // ----------------------------------------------------
    // SIMULAZIONE DELLE TEMPESTE (Ciclo Principale)
    // ----------------------------------------------------
    for (int i = 0; i < num_storms; i++) {

        // Reset del massimo locale prima di ciascuna tempesta
        maximum[i] = -1.0f;
        positions[i] = -1;

        // ====================================================
        // FASE 4.1: BOMBARDAMENTO DI PARTICELLE
        // ====================================================
        for (int j = 0; j < storms[i].size; j++) {
            float energy = (float)storms[i].posval[j * 2 + 1] * 1000.0f; 
            int position = storms[i].posval[j * 2];                

            #pragma omp parallel for schedule(static)
            for (int k = my_start_k; k < my_end_k; k++) {
                update(layer, layer_size, k, position, energy);
            }
        }

        // ====================================================
        // FASE 4.2: RILASSAMENTO (STENCIL)
        // ====================================================
        // Copia parallela dell'array di calcolo
        #pragma omp parallel for schedule(static)
        for (int k = my_start_k; k < my_end_k; k++) {
            layer_copy[k] = layer[k];
        }

        // Buffer per le celle adiacenti dei processi vicini
        float ghost_left_in = 0.0f;
        float ghost_right_in = 0.0f;

        // Scambio Halo Cells via MPI
        MPI_Sendrecv(&layer_copy[my_start_k], 1, MPI_FLOAT, left_neighbor, 0,
                     &ghost_right_in, 1, MPI_FLOAT, right_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&layer_copy[my_end_k - 1], 1, MPI_FLOAT, right_neighbor, 1,
                     &ghost_left_in, 1, MPI_FLOAT, left_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Calcolo dello Stencil (I bordi esterni k=0 e k=layer_size-1 NON vengono modificati)
        #pragma omp parallel for schedule(static)
        for (int k = my_start_k; k < my_end_k; k++) {
            if (k > 0 && k < layer_size - 1) {
                float prev_val = (k == my_start_k) ? ghost_left_in : layer_copy[k - 1];
                float next_val = (k == my_end_k - 1) ? ghost_right_in : layer_copy[k + 1];

                layer[k] = (prev_val + layer_copy[k] + next_val) / 3.0f;
            }
        }

        // Se necessario, aggiorniamo nuovamente la copia o usiamo direttamente layer per la ricerca
        // Ripetiamo uno scambio rapido per verificare il picco locale con il valore rilassato
        MPI_Sendrecv(&layer[my_start_k], 1, MPI_FLOAT, left_neighbor, 2,
                     &ghost_right_in, 1, MPI_FLOAT, right_neighbor, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&layer[my_end_k - 1], 1, MPI_FLOAT, right_neighbor, 3,
                     &ghost_left_in, 1, MPI_FLOAT, left_neighbor, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // ====================================================
        // FASE 4.3: RICERCA DEL MASSIMO LOCALE E GLOBALE
        // ====================================================
        float local_max_val = -1.0f;
        int local_max_pos = -1;

        #pragma omp parallel
        {
            float thread_max_val = -1.0f;
            int thread_max_pos = -1;

            #pragma omp for schedule(static)
            for (int k = my_start_k; k < my_end_k; k++) {
                // Controlla solo i punti interni (1 <= k < layer_size - 1)
                if (k >= 1 && k < layer_size - 1) {
                    float prev_val = (k == my_start_k) ? ghost_left_in : layer[k - 1];
                    float next_val = (k == my_end_k - 1) ? ghost_right_in : layer[k + 1];

                    // Condizione dei picchi locali
                    if (layer[k] > prev_val && layer[k] > next_val) {
                        if (layer[k] > thread_max_val) {
                            thread_max_val = layer[k];
                            thread_max_pos = k;
                        }
                    }
                }
            }

            // Unione thread OpenMP
            #pragma omp critical
            {
                if (thread_max_val > local_max_val) {
                    local_max_val = thread_max_val;
                    local_max_pos = thread_max_pos;
                }
            }
        }

        // Riduzione Globale MPI tra tutti i processi per la tempesta 'i'
        in_struct.val = local_max_val;
        in_struct.pos = local_max_pos;

        MPI_Allreduce(&in_struct, &out_struct, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        maximum[i] = out_struct.val;
        positions[i] = out_struct.pos;
    }

    // DEALLOCAZIONE MEMORIA
    free(layer);
    free(layer_copy);
}