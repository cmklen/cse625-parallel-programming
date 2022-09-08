CSE 625 Parallel Programming – Project 1 
September 1, 2022 
100 points

Due: September 16 (Friday) midnight (Submit your project report, and your Python notebooks p3.ipynb and p4.ipynb to the Blackboard.) 

Notes

- [1] Submit your project report in the PDF document format.
- [2] Name your project report file like this: `your-last-name`.pdf 

Related Materials

- 01_Introduction PowerPoint 

- Python notebooks, 01_Speedup.ipynb and 02_OptimalPEs.ipynb

- CodeBlocks project, TimerDemo (C++ code timing)

- Python code timing using the magic command %timeit and %time 

- Python Data Science Handbook notebooks, Chapter 2 NumPy and Chapter 4 Matplotlib

 
# Assignments

## 1 (20 points) 
Use the CodeBlocks project, TimerDemo, to test the dot product functions, 
   SequentialDot and ThreadDot (defined in Dot.cpp). To do the test, run the function, 
   hpc_helpersTimer (defined in main.cpp), for two ones-vectors of size = 14,000,000, 
   16,000,000, 16,700,000, 16,777,216 and 17,000,000, respectively. Collect the results 
   into the following table:
table in 
|Vector size|14,000,000|16,000,000|16,700,000|16,777,216|17,000,000
|----|----|----|----|----|----
Seq dot result|-|-|-|-|-|					
Thread dot result|-|-|-|-|-|						
Seq runtime (s)	|-|-|-|-|-|					
Thread runtime (s)|-|-|-|-|-|						

The Thread dot result is accurate for two vectors of size 17,000,000, but the sequential dot result is not accurate for two vectors of size larger than 16,777,216 (for example, 17,000,000). Explain why and propose a way to fix the problem.  

## 2 Use Python to compute dot products.

### 2.1 (10 points) Redo Problem 1 in Python using the following Python code:
 
    # Define dot product function

    # Assume v1 and v2 are 1-D NumPy float32 arrays of equal length

    import numpy as np
    def dot (v1, v2):

    result = np.float32(0)
    for i in range(len(v1)):
        result += v1[i] * v2[i]

    return result

    # Test the dot product function
    v1 = np.ones(17000000, dtype = np.float32)
    v2 = np.ones(17000000, dtype = np.float32)

    %time dot(v1, v2)`

Similarly, collect the results for two ones-vectors of size = 14,000,000, 16,000,000, 16,700,000, 16,777,216 and 17,000,000 into the table as shown in Problem 1.

### 2.2 (10 points) 
Redo Problem 1 in Python using NumPy’s ufunc, multiply, and aggregate 
    function, sum, to compute dot products. List your Python code in your project report.

Similarly, collect the results for two ones-vectors of size = 14,000,000, 16,000,000, 
16,700,000, 16,777,216 and 17,000,000 into the table as shown in Problem 1.

Explain why the dot result for float32 vectors of size 17,000,000 using this method is 
accurate. 

### 2.3 (5 points) 
Discuss and compare the runtime results of Problems 1, 2.1 and 2.2. 


## 3 (30 points) 
Consider the Weak (Efficiency) Scalability Analysis for n = 1024×p as 
   shown on slide 18 of the 01_Introduction PowerPoint. Write a Python notebook 
   called, p3.ipynb, to perform the weak scalability analysis,
 
                   for n = 1024 and p =1, 2, 4, 6, 8, 16,3 2, 64, 126,2 58, 512. 

   Show the plots (as given on slide 18) in your project report. 


## 4 (25 points) Suppose we want to compute the sum of two arrays as shown below:

                    A[i]=B[i]+C[i], for i = 0, 1, 2, . . . , n
 
Use the distributed computing algorithm given in the 01_Introduction PowerPoint (in particular, slide 19) to compute the sum of two arrays. Write a Python notebook called p4.ipynb to perform tasks, 4.1-4.3. For this problem, the notebook, 01_Speedup.ipynb, should be very helpful to write your notebook.

### 4.1 Write a Python function to compute the computing time, T. The following is the 
      signature of the function.
       
     def T (p, n, α, β): 
     # Returns computing time
     # Input Parameters:
     # 	p number of processors (PEs)
     # 	n problem size, i.e., number of array elements
     # 	α unit compute time (alpha)
     # 	β unit communication time (beta)
   
    Write some Python code to test the function.

### 4.2 Write a Python function to compute the Speedup, S. The following is the signature of 
        the function.

         def S (p, n, α, β): 
     # Returns speedup
     # Input Parameters:
     # 	p number of processors (PEs)
     # 	n problem size, i.e., number of array elements
     # 	α unit compute time (alpha)
     # 	β unit communication time (beta)

         Write some Python code to test the function.

 
### 4.3 Plot the Speedup, S, as a function of p (number of PEs) for the following cases:

        	n = 1024, α = 1, and β = 0
 	n = 1024, α = 1, and β = 1
 	n = 1024, α = 1, and β = 2
  	n = 1024, α = 1, and β = 3

       Plot all these cases in a single plot and show the plot in your project report.    