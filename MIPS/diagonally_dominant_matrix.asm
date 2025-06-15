# ===============================================================
# Program: diagonally dominant matrix
# Author: Andrea Di Masi
# Description:
# This program checks whether a 3x3 matrix is diagonally dominant.
# A matrix is called diagonally dominant if, for every row, the absolute value of the diagonal element
# is greater than or equal to the sum of the absolute values of the other elements in the same row.
#
# The matrix is stored in memory in row-major order.
# The program iterates over each row, extracts the diagonal and off-diagonal elements,
# computes their absolute values, and compares them according to the condition.
#
# If the matrix is diagonally dominant, it prints:
# "The matrix is simple dominant"
# Otherwise, it prints:
# "The matrix is not simple dominant"

.data
	matrix: .word 	4, -1, 0,
			2,  5, 1,
			20,  50, 100,
	nrows: .word 3
	ncols: .word 3
	msg_dom: .asciiz "The matrix is simple dominant"
	msg_not_dom: .asciiz "The matrix is not simple dominant"
	
.text
.globl main
main:
	la $t0, matrix			# base address matrix
	lw $t1, nrows			# load number of rows
	lw $t2, ncols			# load number of columns
	li $t3, 0			# row index = 0

row_loop:
	# load diagnoal element A
	mul $t4, $t2, $t3		# t4 = row * ncols
	add $t4, $t4, $t3		# t4 = row*ncols + col
	mul $t4, $t4, 4 		# * dimension
	add $t4, $t4, $t0		# address = base + offset
	lw $t5, ($t4)			# load element
	abs $t5, $t5			# absolute value in t5
	
	li $t7, 0			# col index = 0
	li $t8, 0			# sum of off-diagonal = 0

col_loop:
	beq $t7, $t3, skip_diagonal	# if row index = col index
	# load A[row][col]
	mul $t9, $t2, $t3		# t9 = row * ncols
	add $t9, $t9, $t7		# t9 = row*ncols + col
	mul $t9, $t9, 4			# * dimension
	add $t9, $t9, $t0		# address = base + offset
	lw $t6, ($t9)			# load element
	abs $t6, $t6			# absolute value in t5
	add $t8, $t8, $t6		# accumulate

skip_diagonal:
	add $t7, $t7,1			# increase del col index
	blt $t7, $t2, col_loop		# repeat fo all columns
	
	bgt $t8, $t5, not_dominant	# continue with the row_loop
	add $t3, $t3, 1			# next row
	blt $t3, $t1, row_loop		# continue row loop
	j all_dominant	

	
not_dominant:
	la $a0, msg_not_dom		# else the matrix is not simple diagonal
	li $v0, 4			# printing service
	syscall
	li $v0, 10
	syscall

all_dominant:
	la $a0, msg_dom			# otherwise the matrix has a simple dominance
	li $v0, 4			# printing service
	syscall
	li $v0, 10			# termination service
	syscall


			