import java.util.Scanner;
public class PredictGPA {
    public static void main (String [] args)
    {
        Scanner scnr = new Scanner(System.in);
        System.out.println("Semester GPA Predictor: Let's see what your GPA could be this semester...");
        System.out.println(" ");

        // identify variables
        int numHours;
        char letterGrade;
        int totalHours;
        double totalGPA;

        // collect data and sum sum variables numHours and totalGPA
        System.out.print("Course 1: Number of hours? ");
        numHours = scnr.nextInt();
        System.out.print("Course 1: Expected grade? ");
        letterGrade = scnr.next().charAt(0);
        switch (letterGrade){
            case 'A' :
                letterGrade = 4;
                break;
            case 'B' :
                letterGrade = 3;
                break;
            case 'C' :
                letterGrade = 2;
                break;
            case 'D' :
                letterGrade = 1;
                break;
            default :
                letterGrade = 0;

        }
        totalHours = numHours;
        totalGPA = numHours * letterGrade;
        System.out.println(" ");

        System.out.print("Course 2: Number of hours? ");
        numHours = scnr.nextInt();
        System.out.print("Course 2: Expected grade? ");
        letterGrade = scnr.next().charAt(0);
        switch (letterGrade){
            case 'A' :
                letterGrade = 4;
                break;
            case 'B' :
                letterGrade = 3;
                break;
            case 'C' :
                letterGrade = 2;
                break;
            case 'D' :
                letterGrade = 1;
                break;
            default :
                letterGrade = 0;

        }
        totalHours = totalHours + numHours;
        totalGPA = totalGPA + (numHours * letterGrade);
        System.out.println(" ");

        System.out.print("Course 3: Number of hours? ");
        numHours = scnr.nextInt();
        System.out.print("Course 3: Expected grade? ");
        letterGrade = scnr.next().charAt(0);
        switch (letterGrade){
            case 'A' :
                letterGrade = 4;
                break;
            case 'B' :
                letterGrade = 3;
                break;
            case 'C' :
                letterGrade = 2;
                break;
            case 'D' :
                letterGrade = 1;
                break;
            default :
                letterGrade = 0;

        }
        totalHours = totalHours + numHours;
        totalGPA = totalGPA + (numHours * letterGrade);
        System.out.println(" ");

        System.out.print("Course 4: Number of hours? ");
        numHours = scnr.nextInt();
        System.out.print("Course 4: Expected grade? ");
        letterGrade = scnr.next().charAt(0);
        switch (letterGrade){
            case 'A' :
                letterGrade = 4;
                break;
            case 'B' :
                letterGrade = 3;
                break;
            case 'C' :
                letterGrade = 2;
                break;
            case 'D' :
                letterGrade = 1;
                break;
            default :
                letterGrade = 0;

        }
        totalHours = totalHours + numHours;
        totalGPA = totalGPA + (numHours * letterGrade);

        System.out.println(" ");

        // calculate GPA
        totalGPA = totalGPA / totalHours;
        System.out.printf("Your semester GPA would be: %.2f\n", totalGPA);
    }
}
