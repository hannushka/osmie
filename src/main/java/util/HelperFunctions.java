package util;

import org.openstreetmap.atlas.geography.atlas.items.Edge;

public class HelperFunctions {

    public static boolean isRoundAbout(Edge edge){
        if (edge.getTag("junction").orElse("").equals("roundabout")) return true;
        return false;
    }

    public static boolean isNoNameTagged(Edge edge){
        if (edge.getTag("noname").orElse("").equals("yes")) return true;
        return false;
    }
}
