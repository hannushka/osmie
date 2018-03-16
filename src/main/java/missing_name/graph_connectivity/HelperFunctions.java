package missing_name.graph_connectivity;

import org.openstreetmap.atlas.geography.atlas.items.Edge;

public class HelperFunctions {

    public static boolean isRoundAbout(Edge edge){
        if (edge.getTag("junction").orElse("").equals("roundabout")) return true;
        return false;
    }
}
