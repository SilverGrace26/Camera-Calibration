#!/bin/bash


RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

IN_BUILT="./build/Camera"
SELF="./build/CameraImp"

#Header
echo -e "${BLUE}=============================================${NC}"
echo -e "${YELLOW}Running Camera Comparison Tests${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

echo -e "${GREEN}1. With OpenCV's standard function${NC}"
echo -e "${BLUE}---------------------------------------------${NC}"
$IN_BUILT
echo -e "${BLUE}---------------------------------------------${NC}"

FIRST_EXIT=$?
if [ $FIRST_EXIT -ne 0 ]; then
    echo -e "${RED}[$(date)] First program failed with exit code $FIRST_EXIT${NC}"
    exit $FIRST_EXIT
fi

echo ""
echo ""

echo -e "${GREEN}2. With Our Implementation${NC}"
echo -e "${BLUE}---------------------------------------------${NC}"
$SELF
echo -e "${BLUE}---------------------------------------------${NC}"

SECOND_EXIT=$?
if [ $SECOND_EXIT -ne 0 ]; then
    echo -e "${RED}[$(date)] Second program failed with exit code $SECOND_EXIT${NC}"
    exit $SECOND_EXIT
fi

echo ""
echo -e "${YELLOW}[$(date)] Both programs were compiled successfully.${NC}"
echo ""
exit 0