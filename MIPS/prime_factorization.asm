# ===============================================================
# Program: Prime Factorization
# Author: Andrea Di Masi
# Description:
# 	This MIPS assembly program reads a positive integer `n` from
# 	user input and computes its prime factorization.
#
#   	The program finds all prime numbers (including duplicates) that
#   	divide 'n', such that their product equals 'n'. These factors are
#   	stored in an array and printed to the console.
#
# Example:
#   	Input:  60
#   	Output: The prime factors of the number 60 are:
#           2, 2, 3, 5,
#
# Notes:
# - The program uses a helper subroutine to compute n/2 (used as a
#   search limit for factor candidates).
# - Another subroutine performs the actual prime factorization.
# - All results are printed in order with commas in between.
# - No error checking is performed for invalid input.
# ===============================================================

.data
	primesArray: .word 0:20       		# array to store up to 20 prime factors
	msg1: .asciiz "The prime factors of the number "
	msg2: .asciiz " are:\n"
	comma: .asciiz ", "

.text
.globl main
main:
   	li $v0, 5                           	# syscall to read integer n
    	syscall
    	move $s0, $v0                       	# store input in $s0

    	jal half_n                         	# call function to compute n / 2
    	move $a2, $v0                      	# store n / 2 in $a2 (max divisor)
    	move $a1, $s0                      	# $a1 = original n
    	la $a0, primesArray                	# $a0 = address of output array

    	jal prime_factors                  	# call prime factorization function
    	move $t0, $v0                      	# $t0 = count of found prime factors

    	# print message: "The prime factors of the number n are:"
    	la $a0, msg1
    	li $v0, 4
    	syscall

    	move $a0, $s0                      	# print the number itself
    	li $v0, 1
    	syscall

    	la $a0, msg2
    	li $v0, 4
    	syscall

    	# loop through primesArray and print each factor
    	li $t1, 0                          	# index = 0
    	la $t2, primesArray                	# base address of array
loop_print:
    	bge $t1, $t0, exit_program         	# if index == total, stop
    	mul $t3, $t1, 4                    	# offset = index * 4 (word size)
    	add $t3, $t3, $t2                  	# compute address
    	lw $a0, 0($t3)                     	# load value at address
    	li $v0, 1
    	syscall

    	la $a0, comma                      	# print comma after value
    	li $v0, 4
    	syscall

    	addi $t1, $t1, 1                   	# index++
    	j loop_print

exit_program:
    	li $v0, 10
    	syscall

# ==== Prime Factorization Subroutine ====

prime_factors:
    	li $t0, 0                          	# $t0 = count of found factors
    	li $t1, 2                          	# $t1 = current candidate divisor p
    	move $t2, $a0                      	# $t2 = base address of output array
    	move $t3, $a1                      	# $t3 = current value of n to divide

loop_factors:
    	rem $t4, $t3, $t1                  	# if n % p != 0 → try next p
    	bnez $t4, next_p

    	# store p as a prime factor
    	mul $t5, $t0, 4                    	# offset in array
    	add $t5, $t5, $t2
    	sw $t1, 0($t5)

    	div $t3, $t3, $t1                  	# n = n / p
    	add $t0, $t0, 1                   	# factor count++
    	j loop_factors                     	# test same p again

next_p:
   	add $t1, $t1, 1                   	# p = p + 1
    	ble $t1, $a2, loop_factors         	# while p <= n/2, continue
    	move $v0, $t0                      	# return: number of factors found
    	jr $ra

# ==== Subroutine to compute n/2 ====

half_n:
    	rem $t2, $s0, 2                    	# check if even
    	beqz $t2, is_even
is_odd:
    	sub $v0, $s0, 1
    	div $v0, $v0, 2
    	jr $ra
is_even:
    	div $v0, $s0, 2
    	jr $ra
