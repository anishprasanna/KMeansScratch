import java.util.Scanner;
public class PredictGPA {
    public static void main (String [] args)
    {
        Scanner scnr = new Scanner(System.in);

        //Declare variables (We were told to implement the code using only the following four variables)
        int nHours;
        int totalHours = 0;
        char letterGrade;
        double totalGPA = 0.00;

        //Ask the user to enter the grade and the credit hours of the first class and start to progressively calculate the GPA
        System.out.println("Semester GPA Predictor: Let’s see what your GPA could be this semester...");
        System.out.println();
        System.out.println("Course 1: Number of hours? ");
        nHours = scnr.nextInt();
        totalHours = totalHours + nHours;
        System.out.println("Course 1: Expected grade? ");
        letterGrade = scnr.next().charAt(0);
        if (letterGrade == 'A'){
            totalGPA = totalGPA + (nHours * 4);
        }
        else if (letterGrade == 'B'){
            totalGPA = totalGPA + (nHours * 3);
        }
        else if (letterGrade == 'C'){
            totalGPA = totalGPA + (nHours * 2);
        }
        else if (letterGrade == 'D'){
            totalGPA = totalGPA + (nHours * 1);
        }


        //Repeat the same process for the second course
        System.out.println();
        System.out.println("Course 2: Number of hours? ");
        nHours = scnr.nextInt();
        totalHours = totalHours + nHours;
        System.out.println("Course 2: Expected grade? ");
        letterGrade = scnr.next().charAt(0);
        if (letterGrade == 'A'){
            totalGPA = totalGPA + (nHours * 4);
        }
        else if (letterGrade == 'B'){
            totalGPA = totalGPA + (nHours * 3);
        }
        else if (letterGrade == 'C'){
            totalGPA = totalGPA + (nHours * 2);
        }
        else if (letterGrade == 'D'){
            totalGPA = totalGPA + (nHours * 1);
        }


        //Repeat the same process for the third course
        System.out.println();
        System.out.println("Course 3: Number of hours? ");
        nHours = scnr.nextInt();
        totalHours = totalHours + nHours;
        System.out.println("Course 3: Expected grade? ");
        letterGrade = scnr.next().charAt(0);
        if (letterGrade == 'A'){
            totalGPA = totalGPA + (nHours * 4);
        }
        else if (letterGrade == 'B'){
            totalGPA = totalGPA + (nHours * 3);
        }
        else if (letterGrade == 'C'){
            totalGPA = totalGPA + (nHours * 2);
        }
        else if (letterGrade == 'D'){
            totalGPA = totalGPA + (nHours * 1);
        }


        //Finally, repeat the same process one last time for the fourth course
        System.out.println();
        System.out.println("Course 4: Number of hours? ");
        nHours = scnr.nextInt();
        totalHours = totalHours + nHours;
        System.out.println("Course 4: Expected grade? ");
        letterGrade = scnr.next().charAt(0);
        if (letterGrade == 'A'){
            totalGPA = totalGPA + (nHours * 4);
        }
        else if (letterGrade == 'B'){
            totalGPA = totalGPA + (nHours * 3);
        }
        else if (letterGrade == 'C'){
            totalGPA = totalGPA + (nHours * 2);
        }
        else if (letterGrade == 'D'){
            totalGPA = totalGPA + (nHours * 1);
        }


        //Conclude the computation and then print the final GPA
        System.out.println();
        totalGPA = totalGPA / totalHours;
        System.out.print("Your semester GPA would be: ");
        System.out.printf("%.2f", totalGPA);





    }

}
