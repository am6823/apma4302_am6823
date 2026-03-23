include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

reaction2d: reaction2d.o
	${CLINKER} -o reaction2d reaction2d.o ${PETSC_LIB}
	${RM} reaction2d.o