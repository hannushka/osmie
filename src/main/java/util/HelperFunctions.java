package util;

import org.openstreetmap.atlas.geography.atlas.items.Edge;

public class HelperFunctions {

    public static boolean isRoundAbout(Edge edge){
        return edge.getTag("junction").orElse("").equals("roundabout");
    }

    public static boolean isNoNameTagged(Edge edge){
        return edge.getTag("noname").orElse("").equals("yes");
    }
}
