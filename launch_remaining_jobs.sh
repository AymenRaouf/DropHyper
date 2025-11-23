# Arguments to pass to the Python script

ARGS=(
    "CoauthorshipDBLP HGNNP"
    "CoauthorshipDBLP UniSAGE"
    "CoauthorshipDBLP UniGCN"
    "CoauthorshipDBLP UniGIN"
    "CoauthorshipDBLP UniGAT"
    "CoauthorshipDBLP HNHN"

    "Cooking200 HGNNP"
    "Cooking200 UniSAGE"
    "Cooking200 UniGCN"
    "Cooking200 UniGIN"
    "Cooking200 UniGAT"
    "Cooking200 HNHN"
    
    "Tencent2k UniGIN"
)

DEPTHS=(
    "1"
    "2"
    "4"
    "8"
    "12"
)


# OAR job submission options
WALLTIME="61:30:00"  # Walltime for each job
NODES=1             # Number of nodes
CPU=1               # Number of CPUs per node

# Loop over arguments and submit jobs
for ARG in "${ARGS[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
        echo "Submitting job for arguments: $ARG $DEPTH"
        
        # Build the command to submit
        #oarsub -l "nodes=${NODES}/core=${CPU},walltime=${WALLTIME}" -n "Job_$ARGS" \
        #       -S "python3 ${PYTHON_SCRIPT} ${ARGS}"
        oarsub -t night -l walltime=${WALLTIME} "./run_complementary.sh ${ARG} ${DEPTH}"
        
        # Optional: sleep to avoid overwhelming the scheduler
        sleep 1
    done
done

ARGS=(
    "CocitationCiteseer UniGAT"
)


DEPTHS=(
    "8"
    "12"
)

for ARG in "${ARGS[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
        echo "Submitting job for arguments: $ARG $DEPTH"
        
        # Build the command to submit
        #oarsub -l "nodes=${NODES}/core=${CPU},walltime=${WALLTIME}" -n "Job_$ARGS" \
        #       -S "python3 ${PYTHON_SCRIPT} ${ARGS}"
        oarsub -t night -l walltime=${WALLTIME} "./run_complementary.sh ${ARG} ${DEPTH}"
        
        # Optional: sleep to avoid overwhelming the scheduler
        sleep 1
    done
done

ARGS=(
    "Tencent2k UniSAGE"
)


DEPTHS=(
    "4"
    "8"
    "12"
)

for ARG in "${ARGS[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
        echo "Submitting job for arguments: $ARG $DEPTH"
        
        # Build the command to submit
        #oarsub -l "nodes=${NODES}/core=${CPU},walltime=${WALLTIME}" -n "Job_$ARGS" \
        #       -S "python3 ${PYTHON_SCRIPT} ${ARGS}"
        oarsub -t night -l walltime=${WALLTIME} "./run_complementary.sh ${ARG} ${DEPTH}"
        
        # Optional: sleep to avoid overwhelming the scheduler
        sleep 1
    done
done


echo "All jobs submitted!"