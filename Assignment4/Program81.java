import java.util.Scanner;
public class PredictGPA {
    public static void main (String [] args)
    {
        Scanner scnr = new Scanner(System.in);
        int numHours;
        int totalHours = 0;
        String letterGrade;
        int totalGPA = 0;

        System.out.println("Semester GPA Predictor: Let’s see what your GPA could be this semester...");
        //ask for class one
        //hours
        System.out.print("Class 1: Number of hours? ");
        numHours = scnr.nextInt();
        totalHours += numHours;
        //expected grade
        System.out.print("Class 1: Expected grade? ");
        letterGrade = scnr.next();
        totalGPA += addToGPA(letterGrade, numHours);

        //ask for class two
        //hours
        System.out.print("Class 2: Number of hours? ");
        numHours = scnr.nextInt();
        totalHours += numHours;
        //expected grade
        System.out.print("Class 2: Expected grade? ");
        letterGrade = scnr.next();
        totalGPA += addToGPA(letterGrade, numHours);

        //ask for class three
        //hours
        System.out.print("Class 3: Number of hours? ");
        numHours = scnr.nextInt();
        totalHours += numHours;
        //expected grade
        System.out.print("Class 3: Expected grade? ");
        letterGrade = scnr.next();
        totalGPA += addToGPA(letterGrade, numHours);

        //ask for class four
        //hours
        System.out.print("Class 4: Number of hours? ");
        numHours = scnr.nextInt();
        totalHours += numHours;
        //expected grade
        System.out.print("Class 4: Expected grade? ");
        letterGrade = scnr.next();
        totalGPA += addToGPA(letterGrade, numHours);

        //calculated expected gpa
        //print expected gpa
        System.out.println("Your semester GPA would be: " + (Math.round(((double)totalGPA/(double)totalHours) * 100.0) / 100.0));

    }

    private static int addToGPA(String letter, int hours)
    {
        switch(letter)
        {
            case "A":
            {
                return 4 * hours;
            }
            case "B":
            {
                return 3 * hours;
            }
            case "C":
            {
                return 2 * hours;
            }
            case "D":
            {
                return hours;
            }
            default:
                return 0;
        }
    }
}
