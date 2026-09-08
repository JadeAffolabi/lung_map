/**
 * Script to convert all polyline annotations to polygons in QuPath.
 * Written for QuPath v0.5 (may work in other versions).
 */

def lineObjects = getAnnotationObjects().findAll {it.getROI().isLine()}
def polygonObjects = []
def selectedObjects = getSelectedObjects() as List
for (lineObject in lineObjects) {
    def line = lineObject.getROI()
    def points = line.getAllPoints()
    if (points.size() <= 2) {
        println "Skipping line with <= 2 points"
        continue
    }
    def polygon = ROIs.createPolygonROI(line.getAllPoints(), line.getImagePlane())
    def polygonObject = PathObjects.createAnnotationObject(polygon, lineObject.getPathClass())
    polygonObject.setName(lineObject.getName())
    polygonObject.setColor(lineObject.getColor())
    polygonObjects << polygonObject
    // Update the selected objects
    if (selectedObjects.remove(lineObject)) {
        selectedObjects << polygonObject
    }
}
removeObjects(lineObjects, true)
addObjects(polygonObjects)
selectObjects(selectedObjects)