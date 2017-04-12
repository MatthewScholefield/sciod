#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=clang
CCC=clang++
CXX=clang++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=CLang-Linux
CND_DLIB_EXT=so
CND_CONF=Release
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/source/FloatVec.o \
	${OBJECTDIR}/source/NeuralNet.o \
	${OBJECTDIR}/source/Node.o \
	${OBJECTDIR}/source/Row.o

# Test Directory
TESTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}/tests

# Test Files
TESTFILES= \
	${TESTDIR}/TestFiles/f1

# Test Object Files
TESTOBJECTFILES= \
	${TESTDIR}/tests/catch.o \
	${TESTDIR}/tests/simpleTests.o

# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libsciod.${CND_DLIB_EXT}

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libsciod.${CND_DLIB_EXT}: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libsciod.${CND_DLIB_EXT} ${OBJECTFILES} ${LDLIBSOPTIONS} -shared -fPIC

${OBJECTDIR}/source/FloatVec.o: source/FloatVec.cpp
	${MKDIR} -p ${OBJECTDIR}/source
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/source/FloatVec.o source/FloatVec.cpp

${OBJECTDIR}/source/NeuralNet.o: source/NeuralNet.cpp
	${MKDIR} -p ${OBJECTDIR}/source
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/source/NeuralNet.o source/NeuralNet.cpp

${OBJECTDIR}/source/Node.o: source/Node.cpp
	${MKDIR} -p ${OBJECTDIR}/source
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/source/Node.o source/Node.cpp

${OBJECTDIR}/source/Row.o: source/Row.cpp
	${MKDIR} -p ${OBJECTDIR}/source
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/source/Row.o source/Row.cpp

# Subprojects
.build-subprojects:

# Build Test Targets
.build-tests-conf: .build-tests-subprojects .build-conf ${TESTFILES}
.build-tests-subprojects:

${TESTDIR}/TestFiles/f1: ${TESTDIR}/tests/catch.o ${TESTDIR}/tests/simpleTests.o ${OBJECTFILES:%.o=%_nomain.o}
	${MKDIR} -p ${TESTDIR}/TestFiles
	${LINK.cc} -o ${TESTDIR}/TestFiles/f1 $^ ${LDLIBSOPTIONS}   


${TESTDIR}/tests/catch.o: tests/catch.cpp 
	${MKDIR} -p ${TESTDIR}/tests
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -I. -std=c++14 -MMD -MP -MF "$@.d" -o ${TESTDIR}/tests/catch.o tests/catch.cpp


${TESTDIR}/tests/simpleTests.o: tests/simpleTests.cpp 
	${MKDIR} -p ${TESTDIR}/tests
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -I. -std=c++14 -MMD -MP -MF "$@.d" -o ${TESTDIR}/tests/simpleTests.o tests/simpleTests.cpp


${OBJECTDIR}/source/FloatVec_nomain.o: ${OBJECTDIR}/source/FloatVec.o source/FloatVec.cpp 
	${MKDIR} -p ${OBJECTDIR}/source
	@NMOUTPUT=`${NM} ${OBJECTDIR}/source/FloatVec.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -O2 -std=c++14 -fPIC  -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/source/FloatVec_nomain.o source/FloatVec.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/source/FloatVec.o ${OBJECTDIR}/source/FloatVec_nomain.o;\
	fi

${OBJECTDIR}/source/NeuralNet_nomain.o: ${OBJECTDIR}/source/NeuralNet.o source/NeuralNet.cpp 
	${MKDIR} -p ${OBJECTDIR}/source
	@NMOUTPUT=`${NM} ${OBJECTDIR}/source/NeuralNet.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -O2 -std=c++14 -fPIC  -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/source/NeuralNet_nomain.o source/NeuralNet.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/source/NeuralNet.o ${OBJECTDIR}/source/NeuralNet_nomain.o;\
	fi

${OBJECTDIR}/source/Node_nomain.o: ${OBJECTDIR}/source/Node.o source/Node.cpp 
	${MKDIR} -p ${OBJECTDIR}/source
	@NMOUTPUT=`${NM} ${OBJECTDIR}/source/Node.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -O2 -std=c++14 -fPIC  -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/source/Node_nomain.o source/Node.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/source/Node.o ${OBJECTDIR}/source/Node_nomain.o;\
	fi

${OBJECTDIR}/source/Row_nomain.o: ${OBJECTDIR}/source/Row.o source/Row.cpp 
	${MKDIR} -p ${OBJECTDIR}/source
	@NMOUTPUT=`${NM} ${OBJECTDIR}/source/Row.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -O2 -std=c++14 -fPIC  -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/source/Row_nomain.o source/Row.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/source/Row.o ${OBJECTDIR}/source/Row_nomain.o;\
	fi

# Run Test Targets
.test-conf:
	@if [ "${TEST}" = "" ]; \
	then  \
	    ${TESTDIR}/TestFiles/f1 || true; \
	else  \
	    ./${TEST} || true; \
	fi

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
