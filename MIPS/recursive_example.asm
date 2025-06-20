# ---------------------------------------------------------
# Author: Andrea Di Masi
# Recursive function: f(x, y, z)
# Base case:
#   If x * y * z == 0 → return 8
# Recursive case:
#   Return x * y * z * f(z, x, y - 1)
# Parameters: 
#   x = $a0, y = $a1, z = $a2
# Result:
#   Returned in $v0
# ---------------------------------------------------------

.data
	# Initial input values
	x: .word 2
	y: .word 3
	z: .word 1

.text
.globl main
main:
	# Load x, y, and z into argument registers ($a0, $a1, $a2)
	lw $a0, x
	lw $a1, y
	lw $a2, z
	
	# Call the recursive function
	jal recursive_function

	# Print the result returned in $v0
	move $a0, $v0		# move the result in a0
	li $v0, 1		# service to print an integer
	syscall		
	
	# Exit program
	li $v0, 10		# service to terminate the program
	syscall

recursive_function:
	# Compute x * y * z
	mul $t0, $a0, $a1	
	mul $t0, $t0, $a2
	# Base case: if x * y * z == 0, return 8
	beqz $t0, base_case	

	# Save return address and intermediate result (x*y*z) on the stack
	subu $sp, $sp, 8	# allocate 8 bytes on the stack
	sw $ra, 0($sp)		
	sw $t0, 4($sp)
	
	# Prepare arguments for recursive call: f(z, x, y - 1)
	add $t0, $a1, -1	# y-1
	move $a1, $a0		# y <-- x
	move $a0, $a2		# x <-- z
	move $a2, $t0		# z <-- y-1 
	
	# Recursive call
	jal recursive_function	
	
	# Restore saved values from the stack
	lw $ra, 0($sp)		
	lw $t0, 4($sp)
	addi $sp, $sp, 8	# deallocate 8 bytes from the stack
	
	# Multiply recursive result by saved x*y*z
	mul $v0, $v0, $t0
	jr $ra
	
	
base_case:
	# Base case: return 8
	li $v0, 8		
	jr $ra		