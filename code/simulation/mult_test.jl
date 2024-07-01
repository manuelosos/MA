A = ones(3,3,3)

B = zeros(3,3)

fill!(B,2)
print(B)

println("Result \n")
B[1,1] = 3
B[1,2] = 4


A.*B