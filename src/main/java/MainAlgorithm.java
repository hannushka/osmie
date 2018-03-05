
public class MainAlgorithm {

    public void run() {
        SpellObject so = new SpellObject("Gadsbøllevej");
        SCorrecter.run(so);
        so.print();
    }

    public static void main(String[] args) {
        new MainAlgorithm().run();
    }
}
