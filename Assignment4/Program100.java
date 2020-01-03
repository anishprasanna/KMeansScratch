import java.util.Scanner;
public class PredictGPA {
    public static void main(String [] args)
    {
        Scanner scnr = new Scanner(System.in);
        System.out.println("Semester GPA Predictor: Let's see what your GPA could be this semester...");
        System.out.println("");

        //Define variables
       double numHours;
       char letterGrade;
       double totalHours;
       double totalGPA;

        //Ask for course hours and expected grade
        System.out.print("Course 1: Number of hours? ");
        numHours = scnr.nextDouble();
        totalHours = numHours;
        System.out.print("Course 1: Expected grade? ");
        letterGrade = scnr.next().charAt(0);
        switch (letterGrade) {
            case 'A':
                totalGPA = 4.00 * numHours;
                break;
            case 'B':
                totalGPA = 3.00 * numHours;
                break;
            case 'C':
                totalGPA = 2.00 * numHours;
                break;
            case 'D':
                totalGPA = 1.00 * numHours;
                break;
            default:
                totalGPA = 0.00 * numHours;
                break;
        }
        System.out.println("");
        System.out.print("Course 2: Number of hours? ");
        numHours = scnr.nextDouble();
        totalHours = totalHours + numHours;
        System.out.print("Course 2: Expected grade? ");
        letterGrade = scnr.next().charAt(0);
        switch (letterGrade) {
            case 'A':
                totalGPA = totalGPA + 4.00 * numHours;
                break;
            case 'B':
                totalGPA = totalGPA + 3.00 * numHours;
                break;
            case 'C':
                totalGPA = totalGPA + 2.00 * numHours;
                break;
            case 'D':
                totalGPA = totalGPA + 1.00 * numHours;
                break;
            default:
                totalGPA = totalGPA + 0.00 * numHours;
                break;
        }
        System.out.println("");
        System.out.print("Course 3: Number of hours? ");
        numHours = scnr.nextDouble();
        totalHours = totalHours + numHours;
        System.out.print("Course 3: Expected grade? ");
        letterGrade = scnr.next().charAt(0);
        switch (letterGrade) {
            case 'A':
                totalGPA = totalGPA + 4.00 * numHours;
                break;
            case 'B':
                totalGPA = totalGPA + 3.00 * numHours;
                break;
            case 'C':
                totalGPA = totalGPA + 2.00 * numHours;
                break;
            case 'D':
                totalGPA = totalGPA + 1.00 * numHours;
                break;
            default:
                totalGPA = totalGPA + 0.00 * numHours;
                break;
        }
        System.out.println("");
        System.out.print("Course 4: Number of hours? ");
        numHours = scnr.nextDouble();
        totalHours = totalHours + numHours;
        System.out.print("Course 4: Expected grade? ");
        letterGrade = scnr.next().charAt(0);
        switch (letterGrade) {
            case 'A':
                totalGPA = totalGPA + 4.00 * numHours;
                break;
            case 'B':
                totalGPA = totalGPA + 3.00 * numHours;
                break;
            case 'C':
                totalGPA = totalGPA + 2.00 * numHours;
                break;
            case 'D':
                totalGPA = totalGPA + 1.00 * numHours;
                break;
            default:
                totalGPA = totalGPA + 0.00 * numHours;
                break;
        }
        totalGPA = totalGPA / totalHours;
        System.out.println("");
        System.out.printf("Your semester GPA would be: %.2f", totalGPA);
    }
}
