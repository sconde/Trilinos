// Copyright 2002 - 2008, 2010, 2011 National Technology Engineering
// Solutions of Sandia, LLC (NTESS). Under the terms of Contract
// DE-NA0003525 with NTESS, the U.S. Government retains certain rights
// in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
// 
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
// 
//     * Neither the name of NTESS nor the names of its contributors
//       may be used to endorse or promote products derived from this
//       software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <gtest/gtest.h>
#include <stk_io/FillMesh.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <test_utils/StkBalanceRunner.hpp>
#include <stk_unit_test_utils/MeshFixture.hpp>
#include <stk_util/parallel/Parallel.hpp>

class AttributeOrdering : public stk::unit_test_util::MeshFixture
{
public:
  AttributeOrdering()
    : balanceRunner(get_comm()),
      inputFile("reverseOrderAttr.exo"),
      outputDir("outputDir")
  {
    balanceRunner.set_filename(inputFile);
    balanceRunner.set_output_dir(outputDir);
    balanceRunner.set_app_type_defaults("sd");
  }

  std::vector<std::string> get_balanced_field_names()
  {
    setup_empty_mesh(stk::mesh::BulkData::NO_AUTO_AURA);
    stk::io::StkMeshIoBroker stkIo(get_comm());

    const std::string balancedFile = outputDir + "/" + inputFile;
    stk::io::fill_mesh_preexisting(stkIo, balancedFile, get_bulk());

    stk::mesh::Part *blockPart = get_meta().get_part("block_17");
    stk::mesh::FieldVector balancedAttrFields = stkIo.get_ordered_attribute_fields(blockPart);

    std::vector<std::string> fieldNames;
    for (stk::mesh::FieldBase* field : balancedAttrFields) {
      fieldNames.push_back(field->name());
    }
    return fieldNames;
  }

protected:
  stk::integration_test_utils::StkBalanceRunner balanceRunner;

private:
  const std::string inputFile;
  const std::string outputDir;
};

TEST_F(AttributeOrdering, reverseOrderPreservedAfterBalance)
{
  balanceRunner.run_end_to_end();

  std::vector<std::string> expectedAttrFieldNames = {"j", "i", "h", "g", "f", "e", "d", "c", "b", "a"};
  std::vector<std::string> balancedAttrFieldNames = get_balanced_field_names();

  EXPECT_EQ(balancedAttrFieldNames, expectedAttrFieldNames);
}
